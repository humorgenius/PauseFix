"""中文断句的用字约束（共享给评分器与提案引擎）。"""

# 行末绝不能出现的虚词：出现即说明词被拆开了
# 用户实例：被/多收了钱、走廊里朝/国王、这也是为/什么
NO_LINE_END = set(
    "被把朝向为因由从跟同和与给让使对就才都也还只很最更太不没别请并而但若如之乎者以及或则且却又再倘假虽即便"
)

# 行首绝不能出现（切在它们之前 = 把词拆开）
NO_LINE_START = set("的了着过地得是之们个吧吗呢啊呀哦嗯")


def bad_end(ch: str) -> bool:
    return ch in NO_LINE_END


def bad_start(ch: str) -> bool:
    return ch in NO_LINE_START
