import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api import AstrBotConfig
from astrbot.core.message.components import Plain

# 配置日志
logger = logging.getLogger("FavourPro")


class FavourProManager:
    """
    好感度、态度与关系管理系统 (FavourPro)
    - 使用AI驱动的状态快照更新，而非增量计算。
    - 数据结构: {"user_id": {"favour": int, "attitude": str, "relationship": str}}
    """

    def __init__(self, data_path: Path, default_state: Optional[Dict[str, Any]] = None, 
                 min_favour: Optional[int] = None, max_favour: Optional[int] = None):
        """
        初始化管理器，使用由插件主类提供的规范化数据路径。
        :param data_path: 插件的数据存储目录。
        :param default_state: 自定义的默认状态，如果不提供则使用内置默认值。
        :param min_favour: 好感度下限，如果提供则会限制好感度范围。
        :param max_favour: 好感度上限，如果提供则会限制好感度范围。
        """
        self.data_path = data_path
        self.min_favour = min_favour
        self.max_favour = max_favour
        # 使用实例变量而非类变量，避免多实例间的状态污染
        self.DEFAULT_STATE = default_state if default_state is not None else {
            "favour": 0, 
            "attitude": "中立", 
            "relationship": "陌生人"
        }
        self._init_path()
        self.user_data = self._load_data("user_data.json")

    def _init_path(self):
        """初始化数据目录"""
        self.data_path.mkdir(parents=True, exist_ok=True)

    def _load_data(self, filename: str) -> Dict[str, Any]:
        """加载用户状态数据"""
        path = self.data_path / filename
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _save_data(self):
        """保存用户状态数据"""
        path = self.data_path / "user_data.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.user_data, f, ensure_ascii=False, indent=2)

    def get_user_state(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """获取用户的状态，如果不存在则返回默认状态"""
        key = f"{session_id}_{user_id}" if session_id else user_id
        return self.user_data.get(key, self.DEFAULT_STATE.copy())

    def update_user_state(self, user_id: str, new_state: Dict[str, Any], session_id: Optional[str] = None):
        """直接更新用户的状态"""
        key = f"{session_id}_{user_id}" if session_id else user_id
        # 确保好感度是整数
        if 'favour' in new_state:
            try:
                favour_value = int(new_state['favour'])
                # 如果配置了范围限制，则进行限制
                if self.min_favour is not None and favour_value < self.min_favour:
                    favour_value = self.min_favour
                if self.max_favour is not None and favour_value > self.max_favour:
                    favour_value = self.max_favour
                new_state['favour'] = favour_value
            except (ValueError, TypeError):
                # 如果转换失败，则保留旧值或默认值
                current_state = self.get_user_state(user_id, session_id)
                new_state['favour'] = current_state.get('favour', self.DEFAULT_STATE['favour'])

        self.user_data[key] = new_state
        self._save_data()


@register("FavourPro", "柯尔", "基于原作者天各一方的插件重构，AI驱动的多维度好感度系统", "1.1.0")
class FavourProPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        # 获取规范的数据目录并传递给 Manager
        data_dir = StarTools.get_data_dir()
        
        # 从配置构建默认状态
        default_state = {
            "favour": self.config.get("initial_favour", 20),
            "attitude": self.config.get("initial_attitude", "中立"),
            "relationship": self.config.get("initial_relationship", "陌生人")
        }
        
        # 获取好感度范围配置
        min_favour = self.config.get("min_favour")
        max_favour = self.config.get("max_favour")
        
        self.manager = FavourProManager(data_dir, default_state, min_favour, max_favour)

        # 配置日志级别
        log_level = self.config.get("debug_log_level", "INFO")
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO) if isinstance(log_level, str) else log_level)

        self.block_pattern = re.compile(
            r"\s*\[(?=[^\]]*(?:Favour|Attitude|Relationship|F\s*:|A\s*:|R\s*:))[^\]]*\]\s*",
            re.IGNORECASE | re.DOTALL
        )

        self.favour_pattern = re.compile(r"Favour:\s*(-?\d+)")

        # Attitude的值，应该一直持续到它后面出现 ", Relationship:" 或者 "]" 为止
        self.attitude_pattern = re.compile(r"Attitude:\s*(.+?)(?=\s*[,，]\s*Relationship:|\])")

        # Relationship的值，就是它后面直到 "]" 之前的所有内容
        self.relationship_pattern = re.compile(r"Relationship:\s*(.+?)(?=\s*\])")

        # 安装全局拦截器（Monkey Patching）
        # 这样可以拦截所有直接调用 Provider.text_chat 和 Context.send_message 的情况
        # 包括 Conversa 等插件的主动回复
        self._install_global_interceptors(context)

    @property
    def session_based(self) -> bool:
        """动态读取session_based配置"""
        return bool(self.config.get("session_based", False))

    def _get_session_id(self, event: AstrMessageEvent) -> Optional[str]:
        """根据配置决定是否返回会话ID"""
        return event.unified_msg_origin if self.session_based else None

    def _install_global_interceptors(self, context: Context):
        """
        安装全局拦截器，通过 Monkey Patching 拦截 Provider.text_chat 和 Context.send_message
        这样可以确保即使插件（如 Conversa）绑过了框架的钩子系统，好感度标签也能被清理
        """
        plugin_self = self  # 保存引用供闭包使用
        
        # ==================== 拦截 Context.send_message ====================
        original_send_message = context.send_message
        
        async def patched_send_message(session, message_chain):
            """包装后的 send_message，在发送前清理好感度标签"""
            try:
                # 清理消息链中的好感度标签
                if message_chain and hasattr(message_chain, 'chain') and message_chain.chain:
                    for comp in message_chain.chain:
                        if isinstance(comp, Plain) and comp.text:
                            original_text = comp.text
                            cleaned_text = plugin_self.block_pattern.sub('', original_text).strip()
                            if cleaned_text != original_text:
                                comp.text = cleaned_text
                                logger.debug("[FavourPro] send_message 拦截器清理了好感度标签")
            except Exception as e:
                logger.warning(f"[FavourPro] send_message 拦截器处理异常: {e}")
            
            # 调用原始方法
            return await original_send_message(session, message_chain)
        
        # 替换方法
        context.send_message = patched_send_message
        logger.info("[FavourPro] 已安装 Context.send_message 全局拦截器")
        
        # ==================== 拦截所有 Provider 的 text_chat ====================
        def wrap_provider_text_chat(provider):
            """为单个 Provider 实例包装 text_chat 方法"""
            if hasattr(provider, '_favourpro_wrapped'):
                return  # 避免重复包装
            
            original_text_chat = provider.text_chat
            
            async def patched_text_chat(*args, **kwargs):
                """包装后的 text_chat，在返回前清理好感度标签"""
                llm_resp = await original_text_chat(*args, **kwargs)
                
                try:
                    if llm_resp:
                        # 清理 completion_text
                        if llm_resp.completion_text:
                            original_text = llm_resp.completion_text
                            cleaned_text = plugin_self.block_pattern.sub('', original_text).strip()
                            if cleaned_text != original_text:
                                llm_resp.completion_text = cleaned_text
                                logger.debug("[FavourPro] text_chat 拦截器清理了 completion_text")
                        
                        # 清理 result_chain
                        if llm_resp.result_chain and llm_resp.result_chain.chain:
                            for comp in llm_resp.result_chain.chain:
                                if isinstance(comp, Plain) and comp.text:
                                    original_text = comp.text
                                    cleaned_text = plugin_self.block_pattern.sub('', original_text).strip()
                                    if cleaned_text != original_text:
                                        comp.text = cleaned_text
                                        logger.debug("[FavourPro] text_chat 拦截器清理了 result_chain")
                except Exception as e:
                    logger.warning(f"[FavourPro] text_chat 拦截器处理异常: {e}")
                
                return llm_resp
            
            provider.text_chat = patched_text_chat
            provider._favourpro_wrapped = True
        
        # 包装所有已存在的 Provider
        try:
            for provider in context.get_all_providers():
                wrap_provider_text_chat(provider)
            logger.info(f"[FavourPro] 已为 {len(context.get_all_providers())} 个 Provider 安装 text_chat 拦截器")
        except Exception as e:
            logger.warning(f"[FavourPro] 安装 Provider 拦截器时出错: {e}")
        
        # 保存包装函数供后续使用（如动态添加的 Provider）
        self._wrap_provider_text_chat = wrap_provider_text_chat
        self._original_send_message = original_send_message

    @filter.on_llm_request()
    async def add_context_prompt(self, event: AstrMessageEvent, req: ProviderRequest):
        """向LLM注入当前的用户状态，并指示其在响应后更新状态"""
        user_id = event.get_sender_id()
        session_id = self._get_session_id(event)

        state = self.manager.get_user_state(user_id, session_id)

        # 构建当前状态提示
        context_prompt = (
            f"<重要：与当前用户的好感度>你与该用户的关系是：{state['relationship']}，"
            f"好感度为 {state['favour']}，"
            f"你对他的印象是：{state['attitude']}。你的回复应严格参考以上内容。</重要：与当前用户的好感度>"
        )

        # 从配置读取instruction_prompt
        instruction_prompt = self.config.get("instruction_prompt", "")
        
        # 注入到 system_prompt
        if instruction_prompt:
            req.system_prompt += f"\n{instruction_prompt}"
        req.system_prompt += f"\n{context_prompt}"

    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        """
        处理LLM响应，解析并更新状态，然后清理特殊标记 (最终鲁棒版)
        逻辑: 查找 -> 清理 -> 解析 -> 更新
        """
        user_id = event.get_sender_id()
        session_id = self._get_session_id(event)
        original_text = resp.completion_text or ""

        # 调试日志：记录函数调用和原始文本
        logger.debug(f"[FavourPro] on_llm_resp 被调用 - 用户: {user_id}, 会话: {session_id}")
        logger.debug(f"[FavourPro] 原始文本长度: {len(original_text)}")
        
        if not original_text:
            logger.debug("[FavourPro] 原始文本为空，直接返回")
            return

        # 1. 查找：使用宽松的 "主模式" 查找状态块
        block_matches = list(self.block_pattern.finditer(original_text))
        logger.debug(f"[FavourPro] 匹配到的状态块数量: {len(block_matches)}")

        # 如果没有找到任何看起来像状态块的东西，就直接返回，什么都不做
        if not block_matches:
            logger.debug("[FavourPro] 未找到状态块，直接返回")
            return

        # 2. 清理：立即从回复中移除所有状态块，确保用户不会看到它们
        cleaned_text = self.block_pattern.sub('', original_text).strip()
        logger.debug(f"[FavourPro] 清理后文本长度: {len(cleaned_text)}")
        
        # 更新 completion_text（这会同步更新 result_chain）
        resp.completion_text = cleaned_text
        
        # 同时确保 result_chain 中的 Plain 组件也被清理
        if resp.result_chain and resp.result_chain.chain:
            for comp in resp.result_chain.chain:
                if isinstance(comp, Plain) and comp.text:
                    comp.text = self.block_pattern.sub('', comp.text).strip()
        
        logger.debug(f"[FavourPro] 已设置 resp.completion_text，长度: {len(resp.completion_text or '')}")

        # 3. 解析：现在，只对我们捕获的最后一个 `block_text` 进行详细解析
        block_text = block_matches[-1].group(0)
        logger.debug("[FavourPro] 解析最后一个状态块")
        
        favour_match = self.favour_pattern.search(block_text)
        attitude_match = self.attitude_pattern.search(block_text)
        relationship_match = self.relationship_pattern.search(block_text)

        logger.debug(f"[FavourPro] 解析结果: Favour={bool(favour_match)}, Attitude={bool(attitude_match)}, Relationship={bool(relationship_match)}")

        # 如果块里连一个有效参数都找不到，那也直接返回
        if not (favour_match or attitude_match or relationship_match):
            logger.warning("[FavourPro] ⚠️ 警告：状态块中未找到任何有效参数！")
            return

        # 4. 更新：获取当前状态，并用解析出的新值覆盖
        current_state = self.manager.get_user_state(user_id, session_id)
        logger.debug(f"[FavourPro] 当前状态: {current_state}")

        if favour_match:
            new_favour = int(favour_match.group(1).strip())
            old_favour = current_state.get('favour', self.manager.DEFAULT_STATE['favour'])
            delta = new_favour - old_favour
            
            # 验证变化幅度
            max_increase = self.config.get("max_increase")
            max_decrease = self.config.get("max_decrease")
            
            if max_increase is not None and delta > max_increase:
                new_favour = old_favour + max_increase
                logger.info(f"[FavourPro] 好感度提升超限 {delta}→{max_increase}，截断为: {new_favour}")
            elif max_decrease is not None and delta < -max_decrease:
                new_favour = old_favour - max_decrease
                logger.info(f"[FavourPro] 好感度下降超限 {delta}→{-max_decrease}，截断为: {new_favour}")
            
            current_state['favour'] = new_favour
            logger.debug(f"[FavourPro] 更新好感度为: {new_favour} (变化: {delta})")
        if attitude_match:
            new_attitude = attitude_match.group(1).strip(' ,')
            current_state['attitude'] = new_attitude
            logger.debug(f"[FavourPro] 更新印象为: {new_attitude[:50]}")
        if relationship_match:
            new_relationship = relationship_match.group(1).strip(' ,')
            current_state['relationship'] = new_relationship
            logger.debug(f"[FavourPro] 更新关系为: {new_relationship[:50]}")

        self.manager.update_user_state(user_id, current_state, session_id)
        logger.debug("[FavourPro] 状态已保存")

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        """
        在消息发送前进行最后的清理，确保状态块不会被发送给用户
        这是一个额外的保险措施
        """
        result = event.get_result()
        if result is None or not result.chain:
            return
        
        # 遍历消息链中的所有组件，清理 Plain 组件中的状态块
        for comp in result.chain:
            if isinstance(comp, Plain) and comp.text:
                original_text = comp.text
                cleaned_text = self.block_pattern.sub('', original_text).strip()
                if cleaned_text != original_text:
                    comp.text = cleaned_text
                    logger.debug("[FavourPro] on_decorating_result 清理了状态块")

    # ------------------- 管理员命令 -------------------

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        """检查事件发送者是否为AstrBot管理员"""
        return event.role == "admin"

    @filter.command("查询好感")
    async def admin_query_status(self, event: AstrMessageEvent, user_id: str):
        """(管理员) 查询指定用户的状态"""
        if not self._is_admin(event):
            yield event.plain_result(self.config.get("admin_permission_denied_msg", "错误：此命令仅限管理员使用。"))
            return

        session_id = self._get_session_id(event)
        state = self.manager.get_user_state(user_id.strip(), session_id)

        response_text = (
            f"用户 {user_id} 的状态：\n"
            f"好感度：{state['favour']}\n"
            f"关系：{state['relationship']}\n"
            f"态度：{state['attitude']}"
        )
        yield event.plain_result(response_text)

    @filter.command("设置好感")
    async def admin_set_favour(self, event: AstrMessageEvent, user_id: str, value: str):
        """(管理员) 设置指定用户的好感度"""
        if not self._is_admin(event):
            yield event.plain_result(self.config.get("admin_permission_denied_msg", "错误：此命令仅限管理员使用。"))
            return

        try:
            favour_value = int(value)
        except ValueError:
            yield event.plain_result("错误：好感度值必须是一个整数。")
            return

        user_id = user_id.strip()
        session_id = self._get_session_id(event)
        current_state = self.manager.get_user_state(user_id, session_id)
        current_state['favour'] = favour_value
        self.manager.update_user_state(user_id, current_state, session_id)

        yield event.plain_result(f"成功：用户 {user_id} 的好感度已设置为 {favour_value}。")

    @filter.command("设置印象")
    async def admin_set_attitude(self, event: AstrMessageEvent, user_id: str, *, attitude: str):
        """(管理员) 设置指定用户的印象。支持带空格的文本。"""
        if not self._is_admin(event):
            yield event.plain_result(self.config.get("admin_permission_denied_msg", "错误：此命令仅限管理员使用。"))
            return

        user_id = user_id.strip()
        attitude = attitude.strip()
        session_id = self._get_session_id(event)
        current_state = self.manager.get_user_state(user_id, session_id)
        current_state['attitude'] = attitude
        self.manager.update_user_state(user_id, current_state, session_id)

        yield event.plain_result(f"成功：用户 {user_id} 的态度已设置为 '{attitude}'。")

    @filter.command("设置关系")
    async def admin_set_relationship(self, event: AstrMessageEvent, user_id: str, *, relationship: str):
        """(管理员) 设置指定用户的关系。支持带空格的文本。"""
        if not self._is_admin(event):
            yield event.plain_result(self.config.get("admin_permission_denied_msg", "错误：此命令仅限管理员使用。"))
            return

        user_id = user_id.strip()
        relationship = relationship.strip()
        session_id = self._get_session_id(event)
        current_state = self.manager.get_user_state(user_id, session_id)
        current_state['relationship'] = relationship
        self.manager.update_user_state(user_id, current_state, session_id)

        yield event.plain_result(f"成功：用户 {user_id} 的关系已设置为 '{relationship}'。")

    @filter.command("重置好感")
    async def admin_reset_user_status(self, event: AstrMessageEvent, user_id: str):
        """(管理员) 重置指定用户的全部状态为默认值"""
        if not self._is_admin(event):
            yield event.plain_result(self.config.get("admin_permission_denied_msg", "错误：此命令仅限管理员使用。"))
            return

        user_id = user_id.strip()
        session_id = self._get_session_id(event)

        # 直接重置为默认状态
        self.manager.update_user_state(user_id, self.manager.DEFAULT_STATE.copy(), session_id)
        
        yield event.plain_result(f"成功：用户 {user_id} 的状态已重置为默认值。")

    @filter.command("重置负面")
    async def admin_reset_negative_favour(self, event: AstrMessageEvent):
        """(管理员) 重置所有好感度为负数的用户状态"""
        if not self._is_admin(event):
            yield event.plain_result(self.config.get("admin_permission_denied_msg", "错误：此命令仅限管理员使用。"))
            return
        
        # 找出所有好感度<0的用户key
        keys_to_reset = [
            key for key, state in self.manager.user_data.items() 
            if state.get('favour', 0) < 0
        ]

        if not keys_to_reset:
            yield event.plain_result("信息：没有找到任何好感度为负的用户。")
            return

        # 遍历并重置
        for key in keys_to_reset:
            self.manager.user_data[key] = self.manager.DEFAULT_STATE.copy()
        
        self.manager._save_data()
        yield event.plain_result(f"成功：已重置 {len(keys_to_reset)} 个好感度为负的用户。")

    @filter.command("重置全部")
    async def admin_reset_all_users(self, event: AstrMessageEvent):
        """(管理员) 重置所有用户的状态数据"""
        if not self._is_admin(event):
            yield event.plain_result(self.config.get("admin_permission_denied_msg", "错误：此命令仅限管理员使用。"))
            return

        user_count = len(self.manager.user_data)
        self.manager.user_data.clear()
        self.manager._save_data()
        
        yield event.plain_result(f"成功：已清空并重置全部 {user_count} 个用户的状态数据。")

    @filter.command("好感排行")
    async def admin_favour_ranking(self, event: AstrMessageEvent, num: str = "10"):
        """(管理员) 显示好感度最高的N个用户"""
        if not self._is_admin(event):
            yield event.plain_result(self.config.get("admin_permission_denied_msg", "错误：此命令仅限管理员使用。"))
            return
        
        try:
            limit = int(num)
            if limit <= 0:
                raise ValueError
        except ValueError:
            yield event.plain_result("错误：排行数量必须是一个正整数。")
            return

        if not self.manager.user_data:
            yield event.plain_result("当前没有任何用户数据。")
            return

        # 按好感度降序排序
        sorted_users = sorted(
            self.manager.user_data.items(),
            key=lambda item: item[1].get('favour', 0),
            reverse=True
        )

        response_lines = [f"好感度 TOP {limit} 排行榜："]
        for i, (user_key, state) in enumerate(sorted_users[:limit]):
            line = (
                f"{i + 1}. 用户: {user_key}\n"
                f"   - 好感: {state['favour']}, 关系: {state['relationship']}, 印象: {state['attitude']}"
            )
            response_lines.append(line)
        
        yield event.plain_result("\n".join(response_lines))

    @filter.command("负好感排行")
    async def admin_negative_favour_ranking(self, event: AstrMessageEvent, num: str = "10"):
        """(管理员) 显示好感度最低的N个用户"""
        if not self._is_admin(event):
            yield event.plain_result(self.config.get("admin_permission_denied_msg", "错误：此命令仅限管理员使用。"))
            return

        try:
            limit = int(num)
            if limit <= 0:
                raise ValueError
        except ValueError:
            yield event.plain_result("错误：排行数量必须是一个正整数。")
            return

        if not self.manager.user_data:
            yield event.plain_result("当前没有任何用户数据。")
            return
            
        # 按好感度升序排序
        sorted_users = sorted(
            self.manager.user_data.items(),
            key=lambda item: item[1].get('favour', 0)
        )
        
        response_lines = [f"好感度 BOTTOM {limit} 排行榜："]
        for i, (user_key, state) in enumerate(sorted_users[:limit]):
            line = (
                f"{i + 1}. 用户: {user_key}\n"
                f"   - 好感: {state['favour']}, 关系: {state['relationship']}, 印象: {state['attitude']}"
            )
            response_lines.append(line)
            
        yield event.plain_result("\n".join(response_lines))

    async def terminate(self):
        """插件终止时，确保所有数据都已保存，并清理全局拦截器"""
        self.manager._save_data()
        
        # 恢复原始的 send_message 方法
        if hasattr(self, '_original_send_message') and self._original_send_message:
            try:
                self.context.send_message = self._original_send_message
                logger.info("[FavourPro] 已恢复 Context.send_message 原始方法")
            except Exception as e:
                logger.warning(f"[FavourPro] 恢复 send_message 失败: {e}")
        
        # 清理 Provider 的包装标记（注意：无法完全恢复，但标记清理后重载插件可以重新包装）
        try:
            for provider in self.context.get_all_providers():
                if hasattr(provider, '_favourpro_wrapped'):
                    delattr(provider, '_favourpro_wrapped')
        except Exception as e:
            logger.warning(f"[FavourPro] 清理 Provider 标记失败: {e}")
