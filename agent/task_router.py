from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from .config import AppConfig


TASK_MODES = {"simple", "standard", "large", "deep"}
TASK_TYPES = {
    "question",
    "code_explanation",
    "document_workflow",
    "bug_fix",
    "feature_development",
    "review",
    "architecture",
    "refactor",
}
TASK_SCALES = {"small", "medium", "large"}
TASK_RISKS = {"low", "medium", "high"}
TASK_MODE_RANK = {"simple": 0, "standard": 1, "large": 2, "deep": 3}
TASK_RISK_RANK = {"low": 0, "medium": 1, "high": 2}
_MAX_ARTIFACT_HINTS = 32
_STICKY_TASK_ROUTE_REASONS = frozenset(
    {
        "artifact-required",
        "directory-artifact-required",
        "word-artifact-required",
        "mutation-request",
        "conditional-mutation",
        "single-validation",
    }
)


@dataclass(frozen=True)
class TaskRoute:
    """A deterministic, serializable classification and execution policy."""

    task_type: str
    scale: str
    risk: str
    mode: str
    score: int
    reasons: tuple[str, ...]
    max_tool_rounds: int
    require_plan: bool
    chunked_context: bool
    failure_count: int = 0
    artifact_hints: tuple[str, ...] = ()
    directory_hints: tuple[str, ...] = ()
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "scale": self.scale,
            "risk": self.risk,
            "mode": self.mode,
            "score": self.score,
            "reasons": list(self.reasons),
            "max_tool_rounds": self.max_tool_rounds,
            "require_plan": self.require_plan,
            "chunked_context": self.chunked_context,
            "failure_count": self.failure_count,
            "artifact_hints": list(self.artifact_hints),
            "directory_hints": list(self.directory_hints),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskRoute":
        """Load v0.9 routes and safely promote a legacy task_strategy mapping."""

        mode = _enum_value(value.get("mode"), TASK_MODES, "standard")
        default_scale = "large" if mode in {"large", "deep"} else "small" if mode == "simple" else "medium"
        default_rounds = {"simple": 4, "standard": 8, "large": 16, "deep": 24}[mode]
        return cls(
            task_type=_enum_value(value.get("task_type"), TASK_TYPES, "question"),
            scale=_enum_value(value.get("scale"), TASK_SCALES, default_scale),
            risk=_enum_value(value.get("risk"), TASK_RISKS, "low"),
            mode=mode,
            score=_bounded_int(value.get("score"), default=0, minimum=0, maximum=10_000),
            reasons=_reasons_from_value(value.get("reasons")),
            max_tool_rounds=_bounded_int(
                value.get("max_tool_rounds"),
                default=default_rounds,
                minimum=1,
                maximum=10_000,
            ),
            require_plan=bool(value.get("require_plan", mode in {"large", "deep"})),
            chunked_context=bool(value.get("chunked_context", mode in {"large", "deep"})),
            failure_count=_bounded_int(value.get("failure_count"), default=0, minimum=0, maximum=10_000),
            artifact_hints=_artifact_hints_from_value(value.get("artifact_hints")),
            directory_hints=_directory_hints_from_value(value.get("directory_hints")),
            schema_version=_bounded_int(value.get("schema_version"), default=1, minimum=1, maximum=10_000),
        )


class TaskRouter:
    """Classify a request locally without consuming a model request."""

    _DEEP_MARKERS = re.compile(
        r"\b(architecture|migration|refactor|security|audit|root cause|end[- ]to[- ]end|"
        r"comprehensive|all issues|deep analysis|large[- ]scale)\b|"
        r"架构|迁移|重构|安全|审计|根因|端到端|全面|所有问题|深度|大规模",
        re.IGNORECASE,
    )
    _LARGE_MARKERS = re.compile(
        r"\b(repository|codebase|workspace|all files|all documents|entire project|long document|dataset|batch)\b|"
        r"仓库|代码库|工作区|所有文件|全部文件|所有文档|全部文档|整个项目|长文档|数据集|批量",
        re.IGNORECASE,
    )
    _ACTION_MARKERS = re.compile(
        r"\b(fix|implement|change|edit|test|debug|review|audit|analy[sz]e|inspect|build|write|"
        r"create|generate|summari[sz]e|aggregate|render|export|refactor|migrate)\b|"
        r"修复|实现|修改|编辑|测试|调试|审查|审计|分析|检查|构建|编写|创建|新建|生成|总结|汇总|导出|重构|迁移",
        re.IGNORECASE,
    )
    _MUTATION_MARKERS = re.compile(
        r"\b(fix|implement|change|edit|debug|build|write|create|generate|render|export|"
        r"refactor|migrat(?:e|ion)|delete|remove|deploy)\b|"
        r"修复|实现|修改|编辑|调试|构建|编写|创建|新建|生成|导出|重构|迁移|删除|移除|部署",
        re.IGNORECASE,
    )
    _SIMPLE_MARKERS = re.compile(
        r"^(what is|who is|when is|where is|define|translate)\b|"
        r"^(什么是|谁是|何时是|哪里是|定义|翻译)",
        re.IGNORECASE,
    )
    _EXPLANATION_MARKERS = re.compile(
        r"\b(explain|describe|summarize|read this|how does)\b|解释|说明|概述|总结|怎么看|如何工作",
        re.IGNORECASE,
    )
    _ARCHITECTURE_MARKERS = re.compile(r"\b(?:architecture|system design)\b|架构|系统设计", re.IGNORECASE)
    _REFACTOR_MARKERS = re.compile(r"\b(?:refactor|migration)\b|重构|迁移", re.IGNORECASE)
    _BUG_MARKERS = re.compile(
        r"\b(bug|fix|debug|error|failure|root cause)\b|缺陷|修复|调试|错误|失败|根因",
        re.IGNORECASE,
    )
    _FEATURE_MARKERS = re.compile(
        r"\b(implement|add|feature|build|create|generate|render|export)\b|"
        r"实现|新增|添加|功能|构建|创建|新建|生成|导出",
        re.IGNORECASE,
    )
    _REVIEW_MARKERS = re.compile(
        r"\b(review|audit|inspect|analy[sz]e)\b|审查|审计|检查|分析",
        re.IGNORECASE,
    )
    _DOCUMENT_MARKERS = re.compile(
        r"\b(document|documents|docx|word file|word document|pdf|spreadsheet|report)\b|"
        r"文档|材料|报告|汇总|Word|PDF|表格",
        re.IGNORECASE,
    )
    _ARTIFACT_MARKERS = re.compile(
        r"\b(create|generate|write|render|export|save|output)\b[^.!?;,\n]{0,40}?"
        r"\b(file|document|docx|word|pdf|report|folder|directory)\b"
        r"(?:[^.!?;,\n]{0,40}?\.(?:docx|pdf|xlsx?|csv|md|txt)\b)?|"
        r"\b(?:file|document|docx|word|pdf|report)\b[^.!?;,\n]{0,40}?"
        r"\b(?:(?:should|must|will|shall)\s+be|needs?\s+to\s+be|is\s+to\s+be)\s+"
        r"(?:created|generated|written|rendered|exported|saved|output)\b|"
        r"(?:新建|创建|生成|编写|导出|保存|输出)[^。！？；;，,\n]{0,24}?"
        r"(?:文件夹|目录|文件|文档|Word|PDF|报告)|"
        r"(?:文件|文档|Word|PDF|报告)[^。！？；;，,\n]{0,24}?"
        r"(?:新建|创建|生成|导出|保存|输出)",
        re.IGNORECASE,
    )
    _NO_ARTIFACT_MARKERS = re.compile(
        r"\b(?:(?:do|does|did|will|would|can|could|should|must|may|is|are|was|were|has|have|had)"
        r"\s+not|don't|dont|never|no need to|without)\b[^.!?;,\n]{0,16}"
        r"\b(?:create|generate|write|render|export|save|output)\b|"
        r"(?:不要|无需|不需要|禁止|勿|不(?!但|仅)(?:会|再)?|未(?:曾)?|没有|并未)"
        r".{0,8}(?:新建|创建|生成|编写|导出|保存|输出)"
        r"[^。！？；;，,\n]{0,20}?(?:文件|文档|Word|PDF|报告|文件夹|目录)|"
        r"只(?:需|要)?(?:直接)?回复|直接回复(?:即可|就好)?",
        re.IGNORECASE,
    )
    _NEGATED_ARTIFACT_ACTION_MARKERS = re.compile(
        r"\b(?:(?:do|does|did|will|would|can|could|should|must|may|is|are|was|were|has|have|had)"
        r"\s+not|don't|dont|never|no need to|avoid|without)\b.{0,24}"
        r"\b(?:create|generate|write|render|export|save|output)\b|"
        r"\bno\b.{0,64}\b(?:created|generated|written|rendered|exported|saved|output)\b|"
        r"(?:不要|不得|不应|禁止|无需|不需要|勿|不可|不能|严禁|避免|"
        r"不(?!但|仅)(?:会|再)?|未(?:曾)?|没有|并未).{0,16}(?:新建|创建|生成|编写|导出|保存|输出)",
        re.IGNORECASE,
    )
    _IGNORED_GENERATED_FILE_MARKERS = re.compile(
        r"\b(?:ignore|exclude|skip)\b[^.!?;,\n]{0,40}?\b(?:generated files?|build artifacts?)\b|"
        r"(?:忽略|排除|跳过|不要查看)[^。！？；;，,\n]{0,30}?(?:生成文件|构建产物)",
        re.IGNORECASE,
    )
    _ARTIFACT_CONTEXT_BOUNDARY = re.compile(
        r"(?:[\r\n。！？.!?；;，,]+|但(?:是)?|不过|\bbut\b|\bhowever\b)",
        re.IGNORECASE,
    )
    _QUOTED_TEXT_MARKERS = re.compile(
        r"‘[^’\r\n]{0,240}’|“[^”\r\n]{0,240}”|「[^」\r\n]{0,240}」|"
        r"『[^』\r\n]{0,240}』|`[^`\r\n]{0,240}`|\"[^\"\r\n]{0,240}\""
    )
    _META_ARTIFACT_REFERENCE_MARKERS = re.compile(
        r"\b(?:whether|detect|detection|classif(?:y|ication)|route|routing|false positive|misclassif(?:y|ication)|"
        r"phrase|wording)\b|是否|会不会|识别|判断|路由|误判|词组|措辞|引用",
        re.IGNORECASE,
    )
    _META_ARTIFACT_QUESTION_MARKERS = re.compile(
        r"\b(?:does|do|would|will|can|could|should|is|are)\b[^.!?;\n]{0,96}"
        r"\b(?:create|generate)\b[^.!?;\n]{0,96}\b(?:trigger|classif\w*|detect\w*|route|routing)\b|"
        r"(?:创建|生成)[^。！？；;\n]{0,64}(?:是否|会不会|会否)[^。！？；;\n]{0,64}"
        r"(?:触发|识别|判断|路由|误判)|"
        r"(?:是否|会不会|会否)[^。！？；;\n]{0,64}(?:创建|生成)[^。！？；;\n]{0,64}"
        r"(?:触发|识别|判断|路由|误判)",
        re.IGNORECASE,
    )
    _WORD_ARTIFACT_MARKERS = re.compile(r"\b(?:docx|word file|word document)\b|Word|\.docx", re.IGNORECASE)
    _ARTIFACT_BASENAME_HINT = re.compile(
        r"(?<![\w.-])([\w-]{1,128}\.(?:docx|pdf|xlsx?|csv|md|txt))(?![\w.-])",
        re.IGNORECASE,
    )
    _DIRECTORY_HINT_PATTERNS = (
        re.compile(
            r"\b(?:create|generate)\s+(?:(?:a|an|the)\s+)?(?:new\s+)?"
            r"(?!(?:a|an|the|new|temporary|separate|empty|folder|directory)\b)"
            r"([A-Za-z0-9_.-]{1,128})\s+(?:folder|directory)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:create|generate)\s+(?:(?:a|an|the)\s+)?(?:new\s+)?(?:folder|directory)"
            r"\s+(?:named|called)\s+([A-Za-z0-9_.-]{1,128})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:create|generate)\s+(?:(?:a|an|the)\s+)?(?:new\s+)?(?:folder|directory)\s+"
            r"(?!(?:a|an|the|new|temporary|separate|empty|for|to|in|into|inside|within|under|"
            r"beneath|below|near|at|where|with|and|or|of|named|called|folder|directory)\b)"
            r"([A-Za-z0-9_.-]{1,128})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:新建|创建|生成)\s*(?>(?:(?:一个|一份|新的?)\s*)*)"
            r"(?:名为|叫做?|命名为)\s*([\w.-]{1,128})\s*的?\s*(?:文件夹|目录)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:新建|创建|生成)\s*(?>(?:(?:一个|一份|新的?)\s*)*)"
            r"(?!(?:一个|一份|新的?|的|临时|单独|空的?)\s*(?:文件夹|目录))"
            r"([\w.-]{1,128})\s*(?:文件夹|目录)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:新建|创建|生成)\s*(?>(?:(?:一个|一份|新的?)\s*)*)(?:文件夹|目录)"
            r"(?:名为|叫做?|命名为)\s*([\w.-]{1,128})",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:新建|创建|生成)\s*(?>(?:(?:一个|一份|新的?)\s*)*)(?:文件夹|目录)\s*"
            r"(?!用于|用来|以便|在|于|到|位于|放在|并|和|及|临时|单独|空的?|的(?:\s|$)|文件夹|目录)"
            r"([\w.-]{1,128})",
            re.IGNORECASE,
        ),
    )
    _PDF_ARTIFACT_MARKERS = re.compile(r"\bpdf\b|PDF|\.pdf", re.IGNORECASE)
    _HIGH_RISK_MARKERS = re.compile(
        r"\b(security|credential|secret|token|password|permission|authorization|authentication|"
        r"production|deploy|payment|database schema|destructive|delete|wipe|migration)\b|"
        r"安全|凭据|密钥|令牌|密码|权限|授权|认证|生产|部署|支付|数据库架构|破坏性|删除|清空|迁移",
        re.IGNORECASE,
    )
    _CONDITIONAL_MUTATION_MARKERS = re.compile(
        r"\b(?:only|just)\s+(?:fix|change|modify)\b.{0,60}\b(?:if|when)\b|"
        r"\bif\b.{0,80}\b(?:no|not enough|insufficient)\b.{0,40}\b(?:evidence|bug|issue)\b.{0,60}"
        r"\b(?:do not|don't|avoid)\s+(?:change|modify|edit)\b|"
        r"只(?:修复|修改).{0,30}(?:证据确凿|确认|真实).{0,20}(?:Bug|bug|问题|缺陷)|"
        r"若.{0,30}(?:没有|无|不足).{0,30}(?:证据|Bug|bug|问题|缺陷).{0,30}(?:不要|不应|无需).{0,20}(?:修改|改动|修复)|"
        r"如果.{0,30}(?:没有|无|不足).{0,30}(?:证据|Bug|bug|问题|缺陷).{0,30}(?:不要|不应|无需).{0,20}(?:修改|改动|修复)",
        re.IGNORECASE,
    )
    _CONDITIONAL_EVIDENCE_MARKERS = re.compile(
        r"\bif\b.{0,120}\b(?:no|not enough|insufficient|cannot|can't|fail(?:ed)? to find)\b"
        r".{0,80}\b(?:evidence|bug|issue|defect)\b|"
        r"(?:若|如果|如若).{0,120}(?:没有|无|不足|未找到|找不到|无法证实).{0,80}"
        r"(?:证据|Bug|bug|问题|缺陷)",
        re.IGNORECASE,
    )
    _CONDITIONAL_NO_CHANGE_MARKERS = re.compile(
        r"\bskip\s+(?:the\s+)?implement(?:ation)?\b|"
        r"\b(?:do not|don't|dont|must not|should not|no need to)\b.{0,60}"
        r"\b(?:change|modify|edit|fix|implement)\b|"
        r"跳过\s*[`'\"]?implement(?:ation)?[`'\"]?|"
        r"(?:不要|不应|无需|不需要|不得).{0,40}(?:修改|改动|修复|实现)",
        re.IGNORECASE,
    )
    _CONDITIONAL_POSITIVE_ELSE_MARKERS = re.compile(
        r"\bif\b.{0,80}\b(?:find|found|discover|confirm|prove|reproduce)\w*\b.{0,50}"
        r"\b(?:bug|issue|defect)\b.{0,60}\b(?:fix|change|modify|edit|implement)\b.{0,80}"
        r"\b(?:otherwise|else)\b.{0,50}\b(?:leave|keep)\b.{0,30}\b(?:unchanged|as[- ]is)\b|"
        r"\bif\b.{0,80}\b(?:bug|issue|defect)\b.{0,40}\b(?:is\s+)?"
        r"(?:found|discovered|confirmed|proven|reproduced)\b.{0,60}\b(?:fix|change|modify|edit|implement)\b"
        r".{0,80}\b(?:otherwise|else)\b.{0,50}\b(?:leave|keep)\b.{0,30}\b(?:unchanged|as[- ]is)\b|"
        r"(?:若|如果|如若).{0,80}(?:发现|找到|确认|证实|复现).{0,50}(?:Bug|bug|问题|缺陷)"
        r".{0,50}(?:修复|修改|改动|实现).{0,80}(?:否则|不然).{0,40}"
        r"(?:保持原样|维持原样|不做修改|无需修改|不要修改)",
        re.IGNORECASE,
    )
    _CONDITIONAL_SENTENCE_BOUNDARY = re.compile(r"[\r\n。！？.!?]+")
    _CONDITIONAL_CLAUSE_BOUNDARY = re.compile(r"(?:[；;]+|但(?:是)?|不过|\bbut\b|\bhowever\b)", re.IGNORECASE)
    _CONDITIONAL_INTRO_MARKERS = re.compile(r"\b(?:if|when|unless|only|just)\b|若|如果|如若|只|仅", re.IGNORECASE)
    _CONFIRMED_MUTATION_MARKERS = re.compile(
        r"\b(?:fix|implement|change|modify|edit|repair)\b[^.!?;,\n]{0,60}"
        r"\b(?:(?:already|previously)\s+)?(?:confirmed|proven|known)\b|"
        r"(?:修复|实现|修改|改动)[^。！？；;，,\n]{0,32}(?:已经|已)(?:确认|证实)的?",
        re.IGNORECASE,
    )
    _SINGLE_VALIDATION_MARKERS = re.compile(
        r"\b(?:run|execute)\b[^.!?;,\n]{0,40}\b(?:tests?|checks?|validations?)\b"
        r"[^.!?;,\n]{0,16}\b(?:only once|once only|once)\b|"
        r"\b(?:run|execute)\b[^.!?;,\n]{0,32}\b(?:only once|once only|one (?:static )?check)\b|"
        r"\bonly\s+(?:run|execute)\s+(?:one|a single)\b[^.!?;,\n]{0,32}"
        r"\b(?:check|validation|test)\b|"
        r"(?:只|仅)(?:运行|执行)(?:一次|一遍)[^。！？；;，,\n]{0,40}(?:静态检查|检查|验证|测试)|"
        r"(?:静态检查|检查|验证|测试)[^。！？；;，,\n]{0,24}(?:只|仅)(?:运行|执行)(?:一次|一遍)",
        re.IGNORECASE,
    )
    _PER_SCOPE_VALIDATION_MARKERS = re.compile(
        r"\b(?:per|for\s+each|each)\s+(?:package|module|file|project|directory|workspace)\b|"
        r"(?:每个|各)(?:包|模块|文件|项目|目录|工作区)",
        re.IGNORECASE,
    )

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def route(
        self,
        prompt: str,
        *,
        source_file_count: int = 0,
        file_count: int = 0,
        explicit_mode: str | None = None,
        failure_count: int = 0,
    ) -> TaskRoute:
        text = str(prompt or "").strip()
        failures = _bounded_int(failure_count, default=0, minimum=0, maximum=10_000)
        source_files = _bounded_int(source_file_count, default=0, minimum=0, maximum=10_000_000)
        files = _bounded_int(file_count, default=0, minimum=0, maximum=10_000_000)
        action_text = self._without_meta_quoted_text(text)
        action_text = self._IGNORED_GENERATED_FILE_MARKERS.sub("", action_text)
        action_text = self._NO_ARTIFACT_MARKERS.sub("", action_text)
        action = bool(self._ACTION_MARKERS.search(action_text))
        mutation = bool(self._MUTATION_MARKERS.search(action_text))
        large_signal = bool(self._LARGE_MARKERS.search(text))
        deep_signal = bool(self._DEEP_MARKERS.search(text))
        document_signal = bool(self._DOCUMENT_MARKERS.search(text))
        artifact_evidence = self._requested_artifact_evidence(text)
        artifact_signal = bool(artifact_evidence)
        read_only_document_summary = (
            document_signal and bool(self._EXPLANATION_MARKERS.search(text)) and not mutation and not artifact_signal
        )
        engineering_action = action and not read_only_document_summary

        score = 0
        reasons: list[str] = []
        if len(text) >= 2_000:
            score += 2
            reasons.append("long-request")
        elif len(text) >= 600:
            score += 1
            reasons.append("detailed-request")
        if action:
            score += 1
            reasons.append("project-action")
        if mutation:
            reasons.append("mutation-request")
            if self._conditional_mutation_requested(text):
                reasons.append("conditional-mutation")
        if document_signal:
            reasons.append("document-workflow")
        directory_artifact = artifact_signal and self._requested_directory_artifact(text)
        if artifact_signal:
            score += 2
            reasons.append("artifact-required")
            if directory_artifact:
                reasons.append("directory-artifact-required")
            if any(self._WORD_ARTIFACT_MARKERS.search(item) for item in artifact_evidence):
                reasons.append("word-artifact-required")
        if self._single_validation_requested(text):
            reasons.append("single-validation")
        if large_signal:
            score += 2
            reasons.append("large-scope")
        if deep_signal:
            # A short factual question about a deep topic is not itself a deep
            # engineering task. Mutating and repository-wide requests are.
            score += 3 if engineering_action or mutation or large_signal else 1
            reasons.append("deep-reasoning" if engineering_action or mutation or large_signal else "deep-topic")

        large_source_threshold = _bounded_int(
            self.config.get("runtime.large_project_source_files", 500),
            default=500,
            minimum=1,
            maximum=10_000_000,
        )
        large_file_threshold = _bounded_int(
            self.config.get("runtime.large_project_files", 2_000),
            default=2_000,
            minimum=1,
            maximum=10_000_000,
        )
        if source_files >= large_source_threshold:
            score += 2
            reasons.append("large-codebase")
        elif files >= large_file_threshold:
            score += 1
            reasons.append("many-files")

        task_type = self._task_type(text)
        scale = self._scale(
            text,
            action=action or mutation,
            large_signal=large_signal,
            source_files=source_files,
            files=files,
            source_threshold=large_source_threshold,
            file_threshold=large_file_threshold,
        )
        risk = self._risk(text, task_type=task_type, mutation=mutation, action=engineering_action)
        if risk == "high":
            score += 2
            reasons.append("high-risk")
        if failures >= 2:
            score += 2
            reasons.append("repeated-failure")
        elif failures == 1:
            score += 1
            reasons.append("prior-failure")

        configured_mode = str(explicit_mode or self.config.get("runtime.task_mode", "auto")).strip().lower()
        if configured_mode not in TASK_MODES | {"auto"}:
            configured_mode = "auto"
            reasons.append("invalid-mode-auto")
        if configured_mode in TASK_MODES:
            mode = configured_mode
            reasons.append("configured-mode")
        elif score >= 5 or (risk == "high" and mutation):
            mode = "deep"
        elif scale == "large" or score >= 3:
            mode = "large"
        elif score >= 1:
            mode = "standard"
        elif self._SIMPLE_MARKERS.search(text) or task_type == "code_explanation":
            mode = "simple"
        else:
            mode = "standard"

        if document_signal and (large_signal or artifact_signal):
            scale = "large"
            if not deep_signal and risk != "high":
                mode = "large"
        require_plan = mode in {"large", "deep"} or artifact_signal
        chunked_context = mode in {"large", "deep"}
        return TaskRoute(
            task_type=task_type,
            scale=scale,
            risk=risk,
            mode=mode,
            score=score,
            reasons=tuple(dict.fromkeys(reasons)) or ("default",),
            max_tool_rounds=self._round_limit(mode),
            require_plan=require_plan,
            chunked_context=chunked_context,
            failure_count=failures,
            artifact_hints=self._artifact_hints(artifact_evidence),
            directory_hints=self._directory_hints(text) if directory_artifact else (),
        )

    @classmethod
    def _conditional_mutation_requested(cls, text: str) -> bool:
        sentences = tuple(item.strip()[:800] for item in cls._CONDITIONAL_SENTENCE_BOUNDARY.split(text) if item.strip())
        conditional = False
        for sentence in sentences:
            if cls._CONDITIONAL_MUTATION_MARKERS.search(sentence) or cls._CONDITIONAL_POSITIVE_ELSE_MARKERS.search(
                sentence
            ):
                conditional = True
                continue
            for evidence in cls._CONDITIONAL_EVIDENCE_MARKERS.finditer(sentence):
                window = sentence[evidence.start() : evidence.end() + 240]
                if cls._CONDITIONAL_NO_CHANGE_MARKERS.search(window):
                    conditional = True
                    break
        if not conditional:
            return False
        return not cls._has_unconditional_confirmed_mutation(sentences)

    @classmethod
    def _has_unconditional_confirmed_mutation(cls, sentences: tuple[str, ...]) -> bool:
        """Reject a global skip when an independent clause requires a proven fix."""

        for sentence in sentences:
            clauses = (item.strip() for item in cls._CONDITIONAL_CLAUSE_BOUNDARY.split(sentence))
            for clause in clauses:
                if not clause:
                    continue
                for match in cls._CONFIRMED_MUTATION_MARKERS.finditer(clause):
                    prefix = clause[max(0, match.start() - 32) : match.start()]
                    if re.search(r"(?:\b(?:only|just)\b|只|仅)\s*$", prefix, re.IGNORECASE):
                        continue
                    if cls._CONDITIONAL_INTRO_MARKERS.search(clause[: match.start()]):
                        continue
                    return True
        return False

    @classmethod
    def _requested_artifact_evidence(cls, text: str) -> tuple[str, ...]:
        """Return positive action/object matches without global negation leakage."""

        ignored_spans = tuple(match.span() for match in cls._IGNORED_GENERATED_FILE_MARKERS.finditer(text))
        meta_quoted_spans = cls._meta_quoted_spans(text)
        evidence: list[str] = []
        for match in cls._ARTIFACT_MARKERS.finditer(text):
            if any(match.start() < end and match.end() > start for start, end in ignored_spans):
                continue
            if any(start <= match.start() and match.end() <= end for start, end in meta_quoted_spans):
                continue
            prefix = text[max(0, match.start() - 96) : match.start()]
            prefix = cls._ARTIFACT_CONTEXT_BOUNDARY.split(prefix)[-1]
            suffix = text[match.end() : min(len(text), match.end() + 96)]
            suffix = cls._ARTIFACT_CONTEXT_BOUNDARY.split(suffix, maxsplit=1)[0]
            candidate = f"{prefix}{match.group(0)}{suffix}"
            if cls._IGNORED_GENERATED_FILE_MARKERS.search(candidate):
                continue
            if cls._NEGATED_ARTIFACT_ACTION_MARKERS.search(candidate):
                continue
            if cls._META_ARTIFACT_QUESTION_MARKERS.search(candidate):
                continue
            evidence.append(candidate.strip())
        return tuple(evidence)

    @classmethod
    def _requested_directory_artifact(cls, text: str) -> bool:
        """Return whether a positive artifact match directly targets a directory."""

        ignored_spans = tuple(match.span() for match in cls._IGNORED_GENERATED_FILE_MARKERS.finditer(text))
        meta_quoted_spans = cls._meta_quoted_spans(text)
        for match in cls._ARTIFACT_MARKERS.finditer(text):
            if any(match.start() < end and match.end() > start for start, end in ignored_spans):
                continue
            if any(start <= match.start() and match.end() <= end for start, end in meta_quoted_spans):
                continue
            english_action = str(match.group(1) or "").casefold()
            english_object = str(match.group(2) or "").casefold()
            matched_text = match.group(0).rstrip().casefold()
            english_directory = english_action in {"create", "generate"} and english_object in {
                "folder",
                "directory",
            }
            chinese_directory = matched_text.endswith(("文件夹", "目录")) and bool(
                re.match(r"(?:新建|创建|生成)", matched_text)
            )
            if not english_directory and not chinese_directory:
                continue
            prefix = text[max(0, match.start() - 96) : match.start()]
            prefix = cls._ARTIFACT_CONTEXT_BOUNDARY.split(prefix)[-1]
            suffix = text[match.end() : min(len(text), match.end() + 96)]
            suffix = cls._ARTIFACT_CONTEXT_BOUNDARY.split(suffix, maxsplit=1)[0]
            candidate = f"{prefix}{match.group(0)}{suffix}"
            if cls._IGNORED_GENERATED_FILE_MARKERS.search(candidate):
                continue
            if cls._NEGATED_ARTIFACT_ACTION_MARKERS.search(candidate):
                continue
            if cls._META_ARTIFACT_QUESTION_MARKERS.search(candidate):
                continue
            return True
        return False

    @classmethod
    def _meta_quoted_spans(cls, text: str) -> tuple[tuple[int, int], ...]:
        spans: list[tuple[int, int]] = []
        for match in cls._QUOTED_TEXT_MARKERS.finditer(text):
            context = text[max(0, match.start() - 96) : min(len(text), match.end() + 96)]
            if cls._META_ARTIFACT_REFERENCE_MARKERS.search(context):
                spans.append(match.span())
        return tuple(spans)

    @classmethod
    def _without_meta_quoted_text(cls, text: str) -> str:
        spans = cls._meta_quoted_spans(text)
        if not spans:
            return text
        parts: list[str] = []
        cursor = 0
        for start, end in spans:
            parts.extend((text[cursor:start], " "))
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts)

    @classmethod
    def _single_validation_requested(cls, text: str) -> bool:
        for sentence in cls._CONDITIONAL_SENTENCE_BOUNDARY.split(text):
            if not sentence.strip() or cls._PER_SCOPE_VALIDATION_MARKERS.search(sentence):
                continue
            if cls._SINGLE_VALIDATION_MARKERS.search(sentence):
                return True
        return False

    @classmethod
    def _artifact_hints(cls, evidence: tuple[str, ...]) -> tuple[str, ...]:
        """Return only bounded basenames/extensions suitable for completion gates."""

        hints: list[str] = []
        for item in evidence:
            basenames = [match.group(1) for match in cls._ARTIFACT_BASENAME_HINT.finditer(item)]
            if basenames:
                hints.extend(basenames)
            elif cls._WORD_ARTIFACT_MARKERS.search(item):
                hints.append(".docx")
            elif cls._PDF_ARTIFACT_MARKERS.search(item):
                hints.append(".pdf")
            hints.extend(cls._directory_hints(item))
        return _artifact_hints_from_value(hints)

    @classmethod
    def _directory_hints(cls, text: str) -> tuple[str, ...]:
        hints: list[str] = []
        last_pattern_indexes = {2, 6}
        for index, pattern in enumerate(cls._DIRECTORY_HINT_PATTERNS):
            for match in pattern.finditer(text):
                if index in last_pattern_indexes and text[match.end() :].lstrip()[:1] not in {
                    "",
                    ".",
                    "!",
                    "?",
                    ";",
                    ",",
                    "。",
                    "！",
                    "？",
                    "；",
                    "，",
                    "\n",
                }:
                    continue
                hints.append(match.group(1))
        return _directory_hints_from_value(hints)

    def select(self, prompt: str, **kwargs: Any) -> TaskRoute:
        """Compatibility spelling for callers moving from TaskStrategySelector."""

        return self.route(prompt, **kwargs)

    def _round_limit(self, mode: str) -> int:
        defaults = {"simple": 4, "standard": 8, "large": 16, "deep": 24}
        configured = _bounded_int(
            self.config.get("runtime.max_tool_rounds", 8),
            default=8,
            minimum=1,
            maximum=10_000,
        )
        if mode == "simple":
            rounds = min(defaults[mode], configured)
        elif mode == "standard":
            rounds = configured
        else:
            rounds = max(defaults[mode], configured)
        hard_limit = _bounded_int(
            self.config.get("runtime.max_tool_rounds_hard_limit", 32),
            default=32,
            minimum=1,
            maximum=10_000,
        )
        return max(1, min(rounds, hard_limit))

    @classmethod
    def _task_type(cls, text: str) -> str:
        if cls._ARCHITECTURE_MARKERS.search(text):
            return "architecture"
        if cls._REFACTOR_MARKERS.search(text):
            return "refactor"
        if cls._BUG_MARKERS.search(text):
            return "bug_fix"
        if cls._DOCUMENT_MARKERS.search(text):
            return "document_workflow"
        if cls._FEATURE_MARKERS.search(text):
            return "feature_development"
        if cls._REVIEW_MARKERS.search(text):
            return "review"
        if cls._EXPLANATION_MARKERS.search(text):
            return "code_explanation"
        return "question"

    @classmethod
    def _scale(
        cls,
        text: str,
        *,
        action: bool,
        large_signal: bool,
        source_files: int,
        files: int,
        source_threshold: int,
        file_threshold: int,
    ) -> str:
        if len(text) >= 2_000 or large_signal or source_files >= source_threshold or files >= file_threshold:
            return "large"
        if len(text) >= 600 or action:
            return "medium"
        return "small"

    @classmethod
    def _risk(cls, text: str, *, task_type: str, mutation: bool, action: bool) -> str:
        if cls._HIGH_RISK_MARKERS.search(text) and (mutation or action):
            return "high"
        if mutation or task_type in {"bug_fix", "feature_development", "architecture", "refactor"}:
            return "medium"
        return "low"


def more_capable_task_route(previous: TaskRoute, selected: TaskRoute) -> TaskRoute:
    """Upgrade a resume route without letting a generic continuation replace it."""

    previous_mode = TASK_MODE_RANK.get(previous.mode, 1)
    selected_mode = TASK_MODE_RANK.get(selected.mode, 1)
    if selected_mode > previous_mode:
        return _preserve_task_route_constraints(previous, selected)
    if selected_mode == previous_mode and TASK_RISK_RANK.get(selected.risk, 0) > TASK_RISK_RANK.get(previous.risk, 0):
        return _preserve_task_route_constraints(previous, selected)
    if selected_mode == previous_mode and selected.task_type != "question" and selected.task_type != previous.task_type:
        return _preserve_task_route_constraints(previous, selected)
    if selected_mode == previous_mode and selected.score > previous.score and selected.task_type != "question":
        return _preserve_task_route_constraints(previous, selected)
    if selected.failure_count > previous.failure_count:
        return replace(
            previous,
            score=max(previous.score, selected.score),
            reasons=tuple(dict.fromkeys((*previous.reasons, *selected.reasons))),
            failure_count=selected.failure_count,
            artifact_hints=_artifact_hints_from_value((*previous.artifact_hints, *selected.artifact_hints)),
            directory_hints=_directory_hints_from_value((*previous.directory_hints, *selected.directory_hints)),
            require_plan=previous.require_plan or selected.require_plan,
            chunked_context=previous.chunked_context or selected.chunked_context,
        )
    selected_constraints = tuple(reason for reason in selected.reasons if reason in _STICKY_TASK_ROUTE_REASONS)
    reasons = tuple(dict.fromkeys((*previous.reasons, *selected_constraints)))
    artifact_hints = _artifact_hints_from_value((*previous.artifact_hints, *selected.artifact_hints))
    directory_hints = _directory_hints_from_value((*previous.directory_hints, *selected.directory_hints))
    require_plan = previous.require_plan or selected.require_plan
    chunked_context = previous.chunked_context or selected.chunked_context
    if (
        reasons == previous.reasons
        and artifact_hints == previous.artifact_hints
        and directory_hints == previous.directory_hints
        and require_plan == previous.require_plan
        and chunked_context == previous.chunked_context
    ):
        return previous
    return replace(
        previous,
        reasons=reasons,
        artifact_hints=artifact_hints,
        directory_hints=directory_hints,
        require_plan=require_plan,
        chunked_context=chunked_context,
    )


def _preserve_task_route_constraints(previous: TaskRoute, selected: TaskRoute) -> TaskRoute:
    """Keep original task safety constraints when Resume upgrades the route."""

    retained = tuple(reason for reason in previous.reasons if reason in _STICKY_TASK_ROUTE_REASONS)
    reasons = tuple(dict.fromkeys((*selected.reasons, *retained)))
    # Original output constraints are sticky across Resume.  Keep them first
    # when the bounded route cannot represent every newly mentioned artifact.
    artifact_hints = _artifact_hints_from_value((*previous.artifact_hints, *selected.artifact_hints))
    directory_hints = _directory_hints_from_value((*previous.directory_hints, *selected.directory_hints))
    require_plan = selected.require_plan or previous.require_plan
    chunked_context = selected.chunked_context or previous.chunked_context
    if (
        reasons == selected.reasons
        and artifact_hints == selected.artifact_hints
        and directory_hints == selected.directory_hints
        and require_plan == selected.require_plan
        and chunked_context == selected.chunked_context
    ):
        return selected
    return replace(
        selected,
        reasons=reasons,
        artifact_hints=artifact_hints,
        directory_hints=directory_hints,
        require_plan=require_plan,
        chunked_context=chunked_context,
    )


# A more explicit alias for new runtime code.
monotonic_task_route = more_capable_task_route


def _enum_value(value: Any, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else default


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _reasons_from_value(value: Any) -> tuple[str, ...]:
    items = value if isinstance(value, (list, tuple)) else []
    return tuple(str(item) for item in items if str(item).strip()) or ("restored",)


def _artifact_hints_from_value(value: Any) -> tuple[str, ...]:
    items = value if isinstance(value, (list, tuple)) else []
    concrete: list[str] = []
    extensions: list[str] = []
    for item in items:
        hint = str(item).strip()
        if not hint or len(hint) > 160 or "/" in hint or "\\" in hint:
            continue
        if not re.fullmatch(
            r"(?:\.[a-z0-9]{1,10}|[\w-]{1,128}(?:\.[a-z0-9]{1,10})?)",
            hint,
            re.IGNORECASE,
        ):
            continue
        (extensions if hint.startswith(".") else concrete).append(hint)
    # A concrete basename carries more completion evidence than a generic
    # extension, so it receives the bounded slots first.
    return tuple(dict.fromkeys((*concrete, *extensions)))[:_MAX_ARTIFACT_HINTS]


def _directory_hints_from_value(value: Any) -> tuple[str, ...]:
    items = value if isinstance(value, (list, tuple)) else []
    hints: list[str] = []
    for item in items:
        hint = str(item).strip()
        if not hint or len(hint) > 128 or hint in {".", ".."} or "/" in hint or "\\" in hint:
            continue
        if not re.fullmatch(r"[\w.-]{1,128}", hint, re.IGNORECASE):
            continue
        hints.append(hint)
    return tuple(dict.fromkeys(hints))[:_MAX_ARTIFACT_HINTS]
