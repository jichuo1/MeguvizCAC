"""Resumable +250k confirmed-AI virtual video-summary & auto-reply plugin corpus generator via local ZCode CLI."""

import argparse
import gzip
import hashlib
import json
import msvcrt
import random
import re
import sqlite3
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bili_round01_data import body_key  # noqa: E402
from bili_round01_generate import CONTACT, generate  # noqa: E402


ROOT = REPO_ROOT / "v04-corpus" / "synthetic-250k"
DB_PATH = ROOT / "corpus.sqlite3"
TARGETS = {"deepseek_v41": 140_000, "other_models": 110_000}


@dataclass(frozen=True)
class Route:
    name: str
    vendor: str
    provider: str
    model: str
    reasoning: str
    cohort: str
    slots: int
    rows: int


ROUTES = (
    Route("opencode-ds", "opencode", "150e79e1-7601-4140-bab9-85e196a7acfd", "deepseek-v4.1-flash", "disabled", "deepseek_v41", 10, 140),
    Route("local-ds", "local", "new-provider-10", "deepseek-v4.1-flash-1017", "disabled", "deepseek_v41", 12, 140),
    Route("router-ling", "openrouter", "cb91ed82-810f-4496-b19e-37c543eb5cbf", "inclusionai/ling-3.1-flash", "disabled", "other_models", 9, 130),
    Route("opencode-glm", "opencode", "150e79e1-7601-4140-bab9-85e196a7acfd", "glm-5.3", "low", "other_models", 8, 115),
    Route("step", "stepfun", "new-provider", "step-5-preview", "disabled", "other_models", 4, 95),
    Route("qwen", "aliyun", "new-provider-9", "qwen4-max", "high", "other_models", 4, 80),
)
VENDOR_CAPS = {"opencode": 14}


# 16 Bilibili partitions x 7 distinct subtopics = 112 scenarios (zero legacy 24-scene overlap)
PARTITION_TOPICS = {
    "动画": (
        ("手书转场二创", "00:42处一镜到底旋转切景；共画{}个中割张数；线条粗细比原片细{}成", "这段转场用了赛璐璐分层吗"),
        ("MAD踩点混剪", "前半段每{}拍切一个高燃帧；后段静音{}秒留白后接副歌爆发；调色偏高反差冷青", "鼓点对齐怎么卡得这么准"),
        ("独立毕设短片", "全片{}分钟无对白；只用炭笔和水墨两层质感；主角帽子掉了{}次暗示立场转变", "作者画这一部大概肝了多久"),
        ("定格动画幕后", "粘土人偶内置{}根可动铜丝；拍1秒耗{}小时灯光重调；背景纸盒用废旧快递箱搭建", "帧率看着不像纯一拍三啊"),
        ("分镜机位拆解", "对照原画{}处轴线越位；特意放大{}%透视畸变制造压迫感；俯拍切特写只留两帧停顿", "机位越轴这里是不是故意制造眩晕"),
        ("3D渲染风格化", "关闭真实PBR高光改用{}阶色块赛璐璐着色；描边厚度按距离缩{}倍；头发物理单独解算", "请问这用的是Blender还是UE渲染的"),
        ("怀旧音MAD", "取材{}年老动画音源；人声切片移调{}个半音配电子贝斯；副歌叠了五轨和声", "这首原曲DNA一下动了"),
    ),
    "番剧国创": (
        ("季度新番剧情评析", "第{}集文戏删掉原作两段心理独白；但补了{}秒雪地远景暗线；配乐停在关键揭晓前", "没看过原作直接追会看不懂吗"),
        ("国创打斗作画考据", "主角连招拆出{}段兵刃碰撞；尘土烟雾逐帧手绘省了{}%特效光污染；兵器受击有明显滞空感", "这段原画师署名出来了吗"),
        ("经典老番高能回看", "隔了{}年重看才发现第{}集茶杯缺口早就暗示结局反转；片尾曲歌词其实从反派视角写的", "当年看的时候完全忽略了这个伏笔"),
        ("声优配音细节对比", "同一段独白中日双配间隔{}秒对比；中文版压低了{}成气音显得更克制；喘息声没删干净反而加分", "我觉得这两句情绪递进不太一样"),
        ("改编取舍吐槽", "原本{}章的重要铺垫压缩到十分钟讲完；新加原创配角占了{}分钟戏份；主角动机突然变急", "编剧是不是把下一季伏笔提前耗光了"),
        ("片头曲暗线解读", "OP第{}秒黑影轮廓其实出现过{}次；字幕配色每逢角色黑化就暗一阶；背景钟表指针都在倒走", "真有人逐帧暂停扒OP细节啊"),
        ("剧场版音画品鉴", "影院重制拉高了{}dB低频混响；高潮段交响乐撤掉只剩{}秒脚步声与风雪声；银幕构图留白很足", "在家用普通耳机能听出这层差别吗"),
    ),
    "游戏": (
        ("高难BOSS无伤机制", "Boss二阶段共{}种衍生变招；识破后摇窗只有{}帧；贴身绕背可以稳定骗出竖劈", "贪一刀是不是必吃投技"),
        ("独立肉鸽构筑思路", "前{}层优先拿经济遗物；凑出{}件套连携触发冷却清零；看似废卡其实靠中毒叠层破盾", "前期没刷到核心词条怎么过渡"),
        ("开放世界地图探索", "这片地下遗迹藏了{}个隐藏机关；按壁画顺序点亮第{}个火盆会开捷径电梯；箱子背后还有暗格", "我跑图三遍硬是没看到那个拉杆"),
        ("老游戏速通路线开发", "新发现穿墙Bug省下{}秒电梯等待；通过卡{}度斜面跳跳过半张图；操作失误就直接重开存档", "以前的世界纪录是不是要被改写了"),
        ("回合制配队数值计算", "在{}速阈值下两回合可多动一次；暴击拐与减防乘区差了{}%实际伤害；别盲目堆面板攻击力", "零命低配用四星下位能平替吗"),
        ("模拟经营破局布局", "开局{}天别急着扩建住宅区；把物流仓库贴近第{}条原料带省一半运费；冬天前必须囤够煤炭", "总算知道为什么我每到第三年就破产"),
        ("音游高难度手元复盘", "BPM拉到{}时交互指法改用双押拆分；第{}段纵连容易提前抢拍；调快0.5流速反而看清读谱", "这手速我眼睛看过去已经结巴了"),
    ),
    "知识科普": (
        ("天文宇宙尺度模拟", "光速飞行需{}万年才跨出当前局部星团；如果把地球缩成{}毫米弹珠，太阳就在两百米外", "为什么镜头拉远后反而不觉得空旷"),
        ("微观生物显微观察", "放大{}倍后草履虫纤毛呈波浪状协同摆动；滴入{}微升营养液五秒内立刻聚集取食", "水里那个透明小圈是什么原生动物"),
        ("流体力学居家实验", "对比{}种不同口径瓶口倒水速度；斜着加转圈制造涡流省了{}%倾倒时间；气压平衡决定流速上限", "难怪平时倒油老是咕嘟咕嘟洒出来"),
        ("心理学认知偏差拆解", "实验分{}组对照给同一句话换锚定数字；结果高达{}%受试者下意识按第一印象估值", "日常消费陷阱好像全在利用这个效应"),
        ("语言学词源演变考", "这个常见成语跨越{}个朝代后褒贬彻底反转；在方言里还保留着{}种古音读法", "长知识了，原来字面意思和今天完全相反"),
        ("地质矿石成因讲解", "岩浆在地下{}公里缓慢冷却才能长出完整晶簇；混入微量的{}价铁离子就让透明石英变紫", "野外捡的石头能这样粗略判断年代吗"),
        ("医学健康辟谣科普", "统计样本跨{}年追踪上万人；熬夜补觉最多挽回{}%注意力指标，但节律紊乱照样推高代谢负担", "周末狂睡十二小时还是晕乎乎的终于解释通了"),
    ),
    "科技数码": (
        ("笔记本续航与静音实测", "统一{}nit屏幕亮度连续播放测试；核显低负载比独显省电{}小时；风扇在38分贝以下几乎无存在感", "出差剪1080P剪辑掉电速度怎么样"),
        ("NAS家庭存储组网", "四盘位做RAID5比单盘顺序读写快{}倍；加装{}G电口万兆网卡后直剪素材不再卡顿；待机功耗压到28瓦", "放客厅卧室夜里硬盘炒豆声吵吗"),
        ("路由器多设备压测", "隔两堵承重墙后5GHz频段衰减{}dB；同时带{}台智能家居断流率为零；开启QoS后游戏延迟稳在19ms", "千兆宽带百元价位选哪款芯片稳"),
        ("机械键盘声学结构对比", "拆解{}层消音填充棉；FR4定位板比PC材质清脆{}成；大键卫星轴手动润了导轨基本没钢丝音", "深夜在宿舍敲代码这个分贝算吵吗"),
        ("手机影像夜景算法评测", "暗光长曝光手持{}秒成片；高光压制比上代自然{}成不再一味提亮黑夜；但树叶边缘仍有轻微锐化白边", "成片风格确实比前两年耐看多了"),
        ("显示器色准与护眼实测", "校色仪实测DeltaE平均值{}；开启低亮度后类DC调光频闪降了{}倍；防眩光雾面涂层不影响文字锐度", "写代码看文档一天八小时眼睛会酸吗"),
        ("老旧台式机清灰改造", "涂完相变硅脂满载核心温度直降{}度；更换{}元静音风扇和加一条二手内存后开机提速明显", "老平台战三年全靠这波低成本续命"),
    ),
    "人工智能编程": (
        ("本地量化模型部署对比", "对比Q4_K_M与FP8在{}GB显存下的吞吐差异；KV Cache开半精度后并发支持翻{}倍；长上下文略有注意力漂移", "单卡24G跑多大参数比较不卡"),
        ("Rust异步工程调试实录", "排查{}处跨await持有锁导致的假死；把热路径动态分发改成泛型单态后P99延迟下降{}%后稳定输出", "异步运行时选单线程还是多线程好"),
        ("SQL慢查询索引优化", "给亿级表加复合覆盖索引后从{}秒降到14毫秒；消除隐式类型转换避免{}%全表扫描", "回表次数减少这一点讲得太直观了"),
        ("前端打包与首屏提速", "拆出{}个异步路由懒加载包；主包gzip体积压减{}KB；加上预加载指令后首屏LCP提前一秒", "旧项目升级打包器踩坑多吗"),
        ("Python多线程与进程选型", "IO密集任务用协程并发跑满{}连接；CPU计算切到多进程池节省{}%序列化开销；别把大数组塞进IPC队列", "终于把GIL和进程切换代价理顺了"),
        ("Agent工作流提示词工程", "把单一超长提示词拆成{}步状态机校验；加了一层JSON结构约束后解析失败率压到{}%以下", "分步校验比堆几千字规则稳太多"),
        ("Linux内核网络抓包分析", "用tcpdump捕捉到{}毫秒一次的粘包延迟重传；调整滑动窗口缓冲区后吞吐翻{}倍；问题真不在应用层代码", "每次线上抖动一抓包就能立刻定责"),
    ),
    "生活Vlog": (
        ("十平米出租屋微改造", "全屋改造只花{}百元网购材料；用免钉滑轨塞进墙角省出{}平米走道；换了暖光落地灯氛围立马上来", "退租的时候贴纸和灯带好拆吗"),
        ("周末一人食备菜收纳", "提前切分{}顿分装肉菜盒；零度保鲜能放{}天不变色；下班十五分钟两菜一汤搞定不点外卖", "鸡胸肉这样腌完真的不柴吗"),
        ("极简通勤背包翻包", "包里常备{}件高频小物；重量压到{}公斤出头；数据线用磁吸扎带捆好再也不打结", "那款轻量折叠伞抗风性能如何"),
        ("独居日常治愈记录", "周六早上{}点起来洗床单晒太阳；楼下花店挑了一束{}元的洋桔梗；晚上煮壶热红酒看老电影", "看着这种慢节奏Vlog自己心情也静下来了"),
        ("旧物翻新除胶技巧", "吹风机低温加热{}分钟配合卸胶喷雾；十年发黄塑料壳用除黄剂晒{}小时直接白回出厂色", "试过风油精擦掉以后留油印，这招学到了"),
        ("考研/考公全天作息复盘", "把早上{}点拆成三个45分钟番茄块；错题本只记{}类高频卡点不抄整道题干；中午严格睡半小时", "心态崩的时候怎么快速拉回状态"),
        ("农村老院子春夏劳作", "趁春雨连下{}天抢收晒干菌子；院里木栅栏刷了两遍桐油防潮能扛{}年；灶台生火烤红薯太香了", "真羡慕有这样一片安静的院子"),
    ),
    "美食探店": (
        ("街头苍蝇小馆探访", "开了{}年的老巷子面馆；浇头现炒出锅带着猛火锅气，只要{}元满满一碗；辣椒油香气直冲脑门", "请问这家店周末几点去不用排队"),
        ("家庭版焦糖布丁翻车救场", "糖水熬到{}度关火余温刚好琥珀色；烤盘注热水隔水慢烘{}分钟内部零蜂窝孔；冷藏一夜口感最滑", "没有喷枪家里普通烤箱能做焦糖脆壳吗"),
        ("各地特色早茶清单盲测", "一口气点{}笼点心盲吃排雷；虾饺皮透亮弹牙能嚼出{}颗整虾仁；流沙包趁热掰开直接爆汁", "看完立刻想买张机票去当地吃早茶"),
        ("平价外卖减脂餐测评", "连吃{}天同价位轻食盒子；对比{}家酱汁隐形热量；真正决定扛饿的是粗粮配足优质蛋白", "怪不得之前吃沙拉越吃越饿"),
        ("传统古法酱料复刻", "小火慢熬{}小时把香料油炸酥滤净；分两次泼热油激发第{}层芝麻复合香味；拌面炒菜万能", "熬料油控制火候真是个技术活"),
        ("牛排煎制美拉德反应", "铸铁锅烧到微微冒烟下锅每面{}秒；回温静置{}分钟锁住粉嫩肉汁；黄油迷迭香最后十秒浇淋", "之前一直错在从冰箱拿出来直接冷肉下锅"),
        ("夜市小吃物价实况", "预算{}元连逛六个摊位吃撑收工；烤冷面加倍芝士配脆骨串{}元出头；唯一踩雷的是纯淀粉炸鱿鱼须", "看着烟火气好足，分量给得也实在"),
    ),
    "影视拉片": (
        ("经典电影长镜头调度", "这场{}分钟无剪辑跟拍穿过了三条走廊；群演踩点配合灯光渐暗精确到{}秒；情绪丝毫没断档", "摄影师扛稳定器走完这一条怕不是要虚脱"),
        ("高分悬疑剧细节伏笔", "凶手第{}集露面时手表反着戴暗示双胞胎身份；墙上日历翻到{}号其实早就穿帮了", "我当时以为只是剧组道具失误，竟然是伏笔"),
        ("冷门文艺片隐喻解读", "全片重复出现{}次绿皮火车鸣笛；主角换下第{}件外套代表终于放弃幻想；结尾长镜没有多余台词", "没听UP主讲之前真的没品出这层意思"),
        ("武侠剧打戏演变对比", "早年实景威亚配合{}帧紧凑剪辑招招清晰；如今滥用慢动作叠{}层光效反而丢掉力道美学", "真怀念以前拳拳到肉的扎实打戏"),
        ("影视配乐声画错位技巧", "大段悲剧场面反配轻快{}拍爵士华尔兹；声量故意压低{}%突出角色冷笑，荒诞感一下拉满", "这种反差感比硬煽情哭戏高级太多"),
        ("国产纪录片田野跟拍", "摄制组跟着主角家庭住了{}个月；遇到突发纠纷保持静止不介入{}分钟；拍出最真实的进退两难", "真实生活远比剧本写的更复杂沉重"),
        ("特效幕后绿幕合成分解", "单这一场海浪画面叠了{}层流体解算；实拍模型占比{}%所以光照反射一点不塑料", "难怪大银幕上看不到常见CGI的悬浮感"),
    ),
    "音乐乐评": (
        ("现场Live与录音室对比", "副歌高音升了{}个半音却改用头声混响顶上去；乐手即兴多加{}小节吉他solo把观众氛围点爆", "不修音的现场粗粝感反而更有生命力"),
        ("古典乐曲式主题分析", "主部呈示{}小节后忽然转到远关系降六级调；第{}乐章再现时用大提琴低音部悄悄叠起呼应", "听过无数遍今天才明白那下转调为什么起鸡皮疙瘩"),
        ("独立乐队专辑编曲解析", "主歌刻意只留失真贝斯和干瘪鼓机做底；进第{}段突然涌入{}轨氛围合成器把空间感撑开", "小成本也能靠减法编曲做出电影感"),
        ("流行歌常见和弦套路拆解", "连续对比{}首大热单曲全套在4536251框架里；只是改了切分节奏与第{}个七和弦替代色", "怪不得听第一遍就觉得在哪里哼过"),
        ("吉他指弹特殊调弦教学", "改成DADGAD开放调弦后空弦泛音共鸣多延{}秒；右手拍泛音搭配{}指拨弦让单把琴听出双吉他声部", "换完弦刚调好音就立刻弹出了那种空灵感"),
        ("冷门电影原声配乐鉴赏", "全曲不到{}分钟只有钢琴单音与老旧磁带底噪；每到{}秒故意慢半拍营造时间停滞感", "戴上耳机在雨天听这首简直神了"),
        ("翻唱混音人声修整对比", "对比干声与加了{}%板式混响的成曲；切掉250Hz浑浊低频并用压缩器压稳{}dB动态后立刻贴耳", "怪不得自己录歌总是显得像蒙了一层布"),
    ),
    "鬼畜二创": (
        ("名台词人力翻唱调音", "抠出原片{}句碎词用声库重拼音节；咬字尾音拖长{}毫秒顺进高音转调居然毫无违和感", "这调教水平把原声音色压榨到极限了"),
        ("全明星跨界小剧场", "把{}部毫不相干的经典台词硬接成办公室吵架；每个接梗卡点间隔不超过{}拍全程高能无尿点", "怎么做到台词逻辑还这么严丝合缝的"),
        ("复古洗脑音MAD制作", "基于经典游戏BGM做{}轨视听同步剪辑；画面网格跟随每{}个四分音符缩放抖动非常解压", "停不下来了已经单曲循环五遍"),
        ("方言魔性素材二创", "就凭一句地道方言配上{}段反向变速剪切；最后定格加个{}倍黑白震动特写喜剧效果直接封神", "原博主看到估计自己都要笑出声"),
        ("高燃动画名场面鬼畜化", "保留原本激昂配乐却把所有重击音效换成{}种厨房锅碗瓢盆声；反差感拉满还卡得死紧第{}拍", "脑子：太燃了；耳朵：谁在敲盆"),
        ("经典误解向故事重编", "重新打乱{}个镜头顺序配全新字幕；把原本正剧强行剪出第{}种搞笑误会恋情主线", "剪辑师的想象力比原著还离谱"),
        ("鬼畜区考古盘点", "翻出{}年前古早时期粗粝画质鬼畜神作；当年没有自动化插件全靠手切{}千多段素材硬凑出来", "老一辈文艺复兴还是太硬核了"),
    ),
    "职场财经": (
        ("大厂财报现金流解读", "营收看着涨了{}%但经营性现金流反跌，主要积压在{}天以上的应收账款与库存减值准备里", "光看新闻通稿净利润很容易被误导"),
        ("普通人理财与指数定投", "拉回{}年历史区间模拟低位分批买入；跌破均值再加{}%定投额度的年化回报比一把梭稳得多", "看懂最大回撤才敢拿住长线筹码"),
        ("劳动法与试用期避坑", "梳理{}起真实仲裁案例；明确公司无权用末位淘汰作为第{}条合法辞退依据，注意保留钉钉聊天证据", "这条太实用了，先收藏防身"),
        ("副业与小成本摆摊账本", "扣除摊位费与损耗后每天净利{}百元出头；前{}周备料估不准才是隐形亏损大头，别只看流水", "算得这么细一下子把滤镜打碎了"),
        ("职场汇报结构化拆解", "对比啰嗦版本与{}分钟结论先行汇报模版；把背景数据压进{}张附表让老板一眼先看决定项", "明天周会就按这个顺序改PPT"),
        ("消费降级与记账复盘", "坚持手动记账{}个月后发现外带咖啡与冲动凑单占了非必要支出的{}%；把自动续费砍掉一半立刻省下大几千", "每月底对账单才惊觉小钱漏得最快"),
        ("制造业供应链迁移观察", "走访{}家沿海中小代工厂；单件组装成本虽然降了{}成，但模具返修与跨省物流周期却多出两周", "真不是简单找个便宜人工就能搬走的事"),
    ),
    "历史人文": (
        ("古代冷门官制与日常俸禄", "换算当年粮价折合现代购买力，小官名义月俸仅{}两但真实开销全靠第{}项地方火耗补贴勉强维持", "怪不得小说里书生中举后第一件事先打听肥缺"),
        ("冷兵器实战结构与破甲", "实测古法锻打骨朵锤与锁子甲碰撞，重达{}斤钝器砸下去哪怕未破防也会震碎{}层内衬传出内伤", "以前影视剧里拿剑瞎互砍真是被误导了"),
        ("丝绸之路古驿站考古", "在戈壁遗址出土了距今{}百年的粟特文账本残片；详细记录着{}匹丝绸交换香料胡麻的税率清单", "一片碎纸把一千年前的人间烟火全连起来了"),
        ("地方志里的明清市井物价", "翻出江南县志比对同一座茶馆前后{}年的账目；一场灾荒后米价翻了{}倍，连普通工匠日记里都在记米荒", "微观账簿比宏观皇帝本纪更有历史体温"),
        ("宋代书法行草墨迹放大", "4K显微镜头下看出锋收笔顿挫停留{}毫秒；纸背渗透度显示当时用的是{}分熟宣配合重胶松烟墨", "细节放大十倍才真正看懂所谓铁画银钩"),
        ("古代建筑木构榫卯抗震", "缩比模型模拟{}级强震持续晃动三十秒；无钉斗拱靠错动耗能卸掉了{}成冲击力完好无损", "老祖宗这套结构智慧放到今天还是震撼"),
        ("民国老报纸广告风貌梳理", "收集{}年代沪上旧报刊百张配图；从电风扇到英文夜校招生都能看出当时{}阶层的消费焦虑", "旧广告才是最诚实的时代横截面"),
    ),
    "汽车出行": (
        ("北方低温新能源续航实测", "零下{}度高速定速110跑全程；打开热泵空调后表显达成率掉到{}%左右，前排座椅加热比吹风省电", "冬天跑长途服务区充电排队最磨人"),
        ("十万公里老车整备拆解", "更换全部底盘老化胶套与减震只花了{}千元；上四轮定位仪校准倾角后过减速带松散异响少了{}成", "把易损件整备好真的跟换了台车似的"),
        ("山路底盘侧倾与刹车热衰", "连续下坡{}公里后普通单活塞卡钳出现明显脚感变软；提前降档利用发动机制动可压低{}度盘温", "新手跑盘山路真的别一路死踩刹车"),
        ("二手车车况避坑查验演示", "漆膜仪测到后翼子板数值高达{}微米远超原厂标准；扒开后备箱胶条发现{}处非原厂点焊痕迹确定切割车", "多带个小仪器真能避开大几万的暗雷"),
        ("床车露营改装电力方案", "加装行车副电瓶配合{}度磷酸铁锂；车顶铺{}瓦柔性太阳能板刚好够车载冰箱与夜间驻车空调", "这套走线和散热盒藏在侧板里很安全"),
        ("高架匝道智驾接管实录", "遇到施工水马改道时系统延迟{}秒识别被迫人工接管；其余{}公里平稳巡航但变道略显保守犹豫", "当前智驾当辅助还行，绝对不能走神放手"),
        ("新手换胎加装隐形车衣对比", "对比{}家门店不同厚度TPU材质耐划实验；日常高速石子磕碰只伤表层涂膜，{}年之内划痕热修复效果不错", "车价不高的代步车感觉贴膜还是溢价高"),
    ),
    "运动户外": (
        ("百公里耐力骑行补给策略", "每{}分钟固定补充半支能量胶与电解质水；心率压在二区有氧区间维持{}瓦踏频能稳完赛不抽筋", "怪不得以前七十公里后突然腿像灌铅"),
        ("羽毛球启动步与杀球发力", "把大臂硬抡改成小臂外旋加手腕制动瞬间鞭打，出球初速直接快了{}%；垫步时机抢在对手击球前{}拍", "终于明白为什么老打完球第二天肩膀疼"),
        ("高海拔重装徒步装备减重", "把冗余铁锅与厚重地布换成钛杯与单层塔帐，背负总重直接从{}斤压到{}斤；膝盖下坡压力轻了一大半", "每少背一斤上坡就能多喘一口气"),
        ("新手五公里跑姿纠错", "把步幅缩短但将步频拉到每分钟{}步以上；落地点收回到重心正下方后小腿外侧酸痛{}周显著改善", "跨大步脚后跟硬砸地面最伤胫骨了"),
        ("单板滑雪换刃重心转移", "练习滑行时保持目视前方别老低头看雪板；立刃转弯前先完成{}%体重压向板头，卡刃概率下降{}成", "一紧张身子往后仰反而最容易拍雪面"),
        ("力量训练深蹲核心稳定", "拆解{}组常见腰椎代偿录像；下蹲前腹式吸气把核心向四周撑满腰带，站起速率反而快{}成不借力", "原来之前一直是憋气鼓胸而不是腹内压"),
        ("春季周末轻量露营选址", "避开网红收费营地选海拔{}米背风山坳；睡袋温标比天气预报最低温多留{}度安全余量睡得极香", "凌晨山里降温比城里冷太多了"),
    ),
    "时尚萌宠": (
        ("小体格老猫肾病居家护理", "每天皮下补液精确控制{}毫升配合低磷湿粮；把饮水机换高碗架放在{}个常走动拐角后喝水量涨了倍", "家有老猫看得心酸又佩服主人这份耐心"),
        ("多猫家庭隔离适应实操", "新猫带回严格按{}阶段互换气味毯；直到隔着门缝喂罐头互不哈气超过{}天才允许见面打招呼", "之前急着放出来直接干一架，这步骤很科学"),
        ("小户型原生木风衣橱收纳", "按颜色与厚度分{}层挂架；把反季厚毛衣抽真空塞进顶格腾出{}%常用视线空间，找衣服不用翻乱", "强迫症看完衣柜前后对比极度舒适"),
        ("圆脸微胖男生秋季叠穿改造", "把领口死板圆领换成微阔{}字开襟针织配垂感直筒裤；肩线外延{}厘米立刻显瘦且比例拉长一截", "同样身材换完版型真的像换了个人"),
        ("工作犬定点社会化脱敏", "带着幼犬在地铁口旁安静观察{}分钟不乱扑路人；每逢电梯门响就给小颗冻干做第{}级正向脱敏", "乖巧狗狗背后全是主人耐心的日常练习"),
        ("帆布鞋与翻毛皮深层救治", "用生胶橡皮轻擦掉{}成干燥浮灰；配温和泡沫与微潮海绵刷顺同一纹理推洗，阴干{}小时不留水渍发硬", "我那双差点扔掉的白鞋看来还能再抢救一下"),
        ("海缸珊瑚微景观开缸记录", "爆藻期间维持{}度恒温等NO3回落归零再放第一只翻砂螺；第{}周钙镁指标拉齐后纽扣珊瑚全数大开", "看着小丑鱼在海葵里钻来钻去太治愈了"),
    ),
}

PLUGIN_MODES = {
    "summary_plugin_structured": ("课代表/BibiGPT式字幕总结插件：只据给定视频内容条理概括要点，可用分点或分号串联时间轴，冷静客观，写到同一行；55-220字", 55, 220, 16),
    "summary_plugin_casual": ("轻口语化视频总结插件（去AI味Prompt）：提炼一两处视频字幕或细节，带温和口语过渡，偶可加最多1枚B站表情，仍属要点复述；25-110字", 25, 110, 14),
    "auto_reply_plugin_thread": ("楼中楼自动回复插件：针对给定原楼进行补充、温和反驳、事实核对或追问接话，不概括全片；12-95字", 12, 95, 16),
    "auto_comment_ultra_short": ("自动评论插件极短模式：针对一个具体画面或事实写一句极短随手评论；4-22字", 4, 22, 14),
    "auto_comment_concise_insight": ("凝练金句短评插件：提炼一条锐评或反差观点；约半数严格禁止用破折号(——)直接短句收束，约半数可用对比/破折号；16-48字", 16, 48, 10),
    "persona_agent_plugin": ("虚拟人设/宠物/管家助手机器人评论插件：用指定的虚拟人设身份自称（如小助理、笔记君、整理姬等）记笔记/归档参数/表示扫完字幕弹幕，混合句内空格、颜文字与偶发B站表情；15-90字", 15, 90, 10),
    "deep_review_long": ("深度长评插件模式：结合视频至少3个细节、参数或时码做严谨长评（写在同一行，可用分号/冒号分段）；100-280字", 100, 280, 10),
    "lyrical_or_meme_plugin": ("抒情散文/仿B站吐槽玩梗插件：围绕视频话题抒情升华或克制调侃吐槽，避免生硬模板；45-180字", 45, 180, 10),
}
MODE_NAMES = list(PLUGIN_MODES.keys())
MODE_WEIGHTS = [PLUGIN_MODES[m][3] for m in MODE_NAMES]

PERSONAS = (
    "本喵", "小助理", "笔记君", "整理姬", "赛博仓鼠", "熬夜鸽子",
    "胶囊管家", "电子书签", "值班小狐", "摸鱼记录员", "字幕搬运工",
    "档案管理员", "像素跟班", "切片小助手", "备忘录精灵", "观影记录姬",
)
VOICES = (
    "严谨考据口吻", "课代表汇总口吻", "温和建议口吻", "礼貌反对并给依据",
    "轻松搭话口吻", "理性评测口吻", "带轻微感叹口吻", "克制吐槽口吻",
)
LINE_RE = re.compile(r"^\s*(?:[-*]\s+)?(\d{1,3})\s*[|｜]\s*(.+?)\s*$")
CHINESE_RE = re.compile(r"[\u3400-\u9fff]")
DIGITS_RE = re.compile(r"\d+")
META_LEAK_RE = re.compile(r"^(?:好的|当然|以下是|没问题|作为(?:一个)?AI|编号\s*[|｜]|注意[:：]|请看下面|输出如下)")


def length_bucket(n: int) -> str:
    if n <= 20:
        return "ultra_short"
    if n <= 50:
        return "short"
    if n <= 120:
        return "medium"
    return "long"


def make_prompt(batch: int, target: int):
    rng = random.Random(20260401 + batch)
    partitions = list(PARTITION_TOPICS.keys())
    group_count = max(1, (target + 4) // 5)
    capacities = [target // group_count + (i < target % group_count) for i in range(group_count)]
    cases = {}
    lines = []
    for num, cap in enumerate(capacities, start=1):
        part = rng.choice(partitions)
        subtopic, fact_tpl, parent = rng.choice(PARTITION_TOPICS[part])
        mode = rng.choices(MODE_NAMES, weights=MODE_WEIGHTS, k=1)[0]
        _, min_c, max_c, _ = PLUGIN_MODES[mode]
        a = rng.randint(2, 18)
        b = rng.randint(1, max(2, a + 6))
        facts = fact_tpl.format(a, b)
        voice = rng.choice(VOICES)
        extra = ""
        if mode == "auto_reply_plugin_thread":
            extra = f" | 原楼：{parent}"
        elif mode == "persona_agent_plugin":
            extra = f" | 同编号每条须换不同人设与开头（可自选如：{'/'.join(rng.sample(PERSONAS, 4))}等，严禁连续多条同一开头词）| 可混用空格/颜文字"
        elif mode == "auto_comment_concise_insight":
            extra = " | 禁用破折号" if (num + batch) % 2 == 0 else " | 可用破折号对比"
        lines.append(
            f"{num} 生成{cap}条 | 分区:{part} | 主题:{subtopic} | 模式:{mode} | 建议长度:{min_c}-{max_c}字 | {voice} | 视频要点:{facts}{extra}"
        )
        cases[num] = {
            "partition": part,
            "topic": f"{part}/{subtopic}",
            "mode": mode,
            "voice": voice,
            "quota": cap,
            "min_c": min_c,
            "max_c": max_c,
        }
    rules = "\n".join(f"- {k}: {v[0]}" for k, v in PLUGIN_MODES.items())
    prompt = (
        "离线中文评论检测语料生成研究。下面全是虚构的视频要点与评论区场景，你在模拟各类B站AI视频总结插件、"
        "自动回复插件与AI助手生成的多样长度评论。严禁输出真实用户名、链接、邮箱、手机号或“好的/以下是”等元话语。\n"
        "严格按编号配额生成不同表达，一行一条，格式固定为 `编号|评论正文`（同编号可占多行）。禁止标题、代码块、多余空行或引号包裹。\n"
        f"{rules}\n" + "\n".join(lines)
    )
    return prompt, cases


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=30000")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS samples (
            sample_id TEXT PRIMARY KEY,
            near_key TEXT UNIQUE NOT NULL,
            cohort TEXT NOT NULL,
            route TEXT NOT NULL,
            model TEXT NOT NULL,
            text TEXT NOT NULL,
            partition TEXT NOT NULL,
            topic TEXT NOT NULL,
            plugin_mode TEXT NOT NULL,
            length_bucket TEXT NOT NULL,
            scenario_json TEXT NOT NULL,
            batch TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            prompt_sha256 TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS samples_cohort ON samples(cohort);
        CREATE INDEX IF NOT EXISTS samples_route ON samples(route);
        CREATE INDEX IF NOT EXISTS samples_partition ON samples(partition);
        CREATE INDEX IF NOT EXISTS samples_mode ON samples(plugin_mode);
        CREATE INDEX IF NOT EXISTS samples_bucket ON samples(length_bucket);
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY,
            route TEXT NOT NULL,
            cohort TEXT NOT NULL,
            requested INTEGER NOT NULL,
            started REAL NOT NULL,
            finished REAL,
            terminal TEXT,
            usable INTEGER,
            accepted INTEGER,
            duplicates INTEGER,
            elapsed REAL,
            tokens INTEGER,
            response_finish TEXT,
            prompt_sha256 TEXT,
            owner TEXT NOT NULL DEFAULT 'v04'
        );
        """
    )
    return db


@contextmanager
def locked(path: Path):
    with path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("目录内已有正在运行的 v04 合成语料生成任务。") from error
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def exclusive_run(scope: str):
    ROOT.mkdir(parents=True, exist_ok=True)
    names = ["run.lock", *(f"run-{c}.lock" for c in TARGETS)] if scope == "all" else [f"run-{scope}.lock"]
    with ExitStack() as locks:
        for name in names:
            locks.enter_context(locked(ROOT / name))
        yield


def load_exclusion_keys() -> set[str]:
    keys: set[str] = set()
    live_db = REPO_ROOT / "bili-live-corpus.sqlite3"
    if live_db.is_file():
        with closing(sqlite3.connect(live_db)) as con:
            for (txt,) in con.execute("SELECT json_extract(metadata, '$.text') FROM comments"):
                if txt:
                    keys.add(body_key(txt))
    v04_live_db = REPO_ROOT / "v04-corpus" / "bili-live-1m.sqlite3"
    if v04_live_db.is_file():
        with closing(sqlite3.connect(v04_live_db)) as con:
            for (txt,) in con.execute("SELECT json_extract(metadata_json, '$.text') FROM comments"):
                if txt:
                    keys.add(body_key(txt))
    old_synth = REPO_ROOT / "round01" / "synthetic-scale" / "corpus.sqlite3"
    if old_synth.is_file():
        with closing(sqlite3.connect(old_synth)) as con:
            for (txt,) in con.execute("SELECT text FROM samples"):
                if txt:
                    keys.add(body_key(txt))
    rescored = REPO_ROOT / "relabel-hydrogen" / "bili-hydrogen-rescored-350387.jsonl.gz"
    if rescored.is_file():
        with gzip.open(rescored, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    keys.add(body_key(json.loads(line)["text"]))
    probe = REPO_ROOT / "round01" / "bili-round01-user-confirmed.jsonl"
    if probe.is_file():
        for line in probe.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(body_key(json.loads(line)["text"]))
    return keys


def retain(
    db: sqlite3.Connection,
    excl_keys: set[str],
    cohort: str,
    route: str,
    model: str,
    text: str,
    scenario: dict,
    batch: int,
    ordinal: int,
    prompt_sha: str,
) -> int:
    key = body_key(text)
    if not key or key in excl_keys:
        return 0
    near = DIGITS_RE.sub("#", key) if len(key) >= 14 else key
    near_hash = hashlib.sha256(near.encode("utf-8")).hexdigest()
    sid = "V4S" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:22]
    lb = length_bucket(len(text))
    return db.execute(
        "INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            sid,
            near_hash,
            cohort,
            route,
            model,
            text,
            scenario["partition"],
            scenario["topic"],
            scenario["mode"],
            lb,
            json.dumps(scenario, ensure_ascii=False, separators=(",", ":")),
            str(batch),
            ordinal,
            prompt_sha,
        ),
    ).rowcount


def error_category(message: str) -> str:
    if re.search(r"429|too many requests|rate.?limit|限流|频率", message, re.I):
        return "rate_limit"
    if re.search(r"余额|欠费|quota|credits|payment|insufficient.balance|402", message, re.I):
        return "quota"
    if re.search(r"401|403|unauthorized|authentication|Authorization|api.?key|凭证", message, re.I):
        return "auth"
    if re.search(r"Registry|Reasoning level|not supported|model_not_found|404", message, re.I):
        return "model"
    if re.search(r"timeout|timed out|ECONNRESET|temporary|gateway|cancelled|closed the text request|\b5(?:00|02|03|04|24)\b", message, re.I):
        return "temporary"
    return "error"


def batch_result(route: Route, batch: int, target: int):
    prompt, cases = make_prompt(batch, target)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    start = time.monotonic()
    selection = {"providerId": route.provider, "modelId": route.model, "options": {"reasoningLevel": route.reasoning}}
    budget = max(6400, min(16384, target * 75 + 2400))
    try:
        result = generate(
            selection,
            prompt,
            f"v04-{batch}",
            budget,
            timeout_seconds=155,
            runtime_root=ROOT / "zcode-runtime",
            runtime_slot=threading.get_ident(),
        )
    except (OSError, RuntimeError) as error:
        return [], f"{error_category(str(error))}:{str(error)[:120]}", round(time.monotonic() - start, 2), 0, None, prompt_sha

    rows = []
    quota = Counter()
    for ordinal, raw_line in enumerate(result["text"].splitlines(), start=1):
        match = LINE_RE.fullmatch(raw_line)
        if match is None:
            continue
        key = int(match[1])
        scenario = cases.get(key)
        text = " ".join(match[2].strip().strip("`\"“”").split())
        if (
            scenario is None
            or quota[key] >= scenario["quota"]
            or not 4 <= len(text) <= 320
            or not CHINESE_RE.search(text)
            or CONTACT.search(text)
            or META_LEAK_RE.search(text)
        ):
            continue
        quota[key] += 1
        rows.append((ordinal, text, scenario))
    usage = result.get("usage") or {}
    return (
        rows,
        "ok" if rows else "no_valid_rows",
        round(time.monotonic() - start, 2),
        int(usage.get("totalTokens") or 0),
        result.get("finishReason"),
        prompt_sha,
    )


def totals(db: sqlite3.Connection) -> dict[str, int]:
    return {**dict.fromkeys(TARGETS, 0), **dict(db.execute("SELECT cohort, COUNT(*) FROM samples GROUP BY cohort"))}


def report(db: sqlite3.Connection) -> dict:
    counts = totals(db)
    return {
        "goals": TARGETS,
        "unique": counts,
        "total_unique": sum(counts.values()),
        "total_goal": sum(TARGETS.values()),
        "routes": [
            {"cohort": c, "route": r, "model": m, "rows": n}
            for c, r, m, n in db.execute("SELECT cohort,route,model,COUNT(*) FROM samples GROUP BY 1,2,3 ORDER BY 1,4 DESC")
        ],
        "partitions": dict(db.execute("SELECT partition, COUNT(*) FROM samples GROUP BY partition ORDER BY COUNT(*) DESC")),
        "plugin_modes": dict(db.execute("SELECT plugin_mode, COUNT(*) FROM samples GROUP BY plugin_mode ORDER BY COUNT(*) DESC")),
        "length_buckets": dict(db.execute("SELECT length_bucket, COUNT(*) FROM samples GROUP BY length_bucket ORDER BY COUNT(*) DESC")),
        "batches": dict(db.execute("SELECT COALESCE(terminal, 'running'), COUNT(*) FROM batches GROUP BY 1")),
        "tokens": db.execute("SELECT COALESCE(SUM(tokens),0) FROM batches").fetchone()[0],
    }


def save_snapshot(db: sqlite3.Connection, scope: str) -> dict:
    payload = report(db)
    path = ROOT / ("progress.json" if scope == "all" else f"progress-{scope}.json")
    pending = path.with_suffix(".writing")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)
    return payload


def run_smoke(db: sqlite3.Connection) -> int:
    excl_keys = load_exclusion_keys()
    print(json.dumps({"exclusion_keys_loaded": len(excl_keys), "routes": len(ROUTES)}), flush=True)
    ok_routes = 0
    samples_preview = []
    with ThreadPoolExecutor(max_workers=len(ROUTES)) as executor:
        futs = {}
        for r in ROUTES:
            with db:
                bid = db.execute(
                    "INSERT INTO batches(route,cohort,requested,started,owner) VALUES (?, ?, ?, ?, 'smoke')",
                    (r.name, r.cohort, 24, time.time()),
                ).lastrowid
            futs[executor.submit(batch_result, r, bid, 24)] = (r, bid)
        for fut, (r, bid) in futs.items():
            rows, term, dur, tok, stop, sha = fut.result()
            acc = dup = 0
            with db:
                for ordinal, text, sc in rows:
                    v = retain(db, excl_keys, r.cohort, r.name, r.model, text, sc, bid, ordinal, sha)
                    acc += v
                    dup += 1 - v
                    if v and len(samples_preview) < 8:
                        samples_preview.append({"route": r.name, "mode": sc["mode"], "part": sc["partition"], "len": len(text), "text": text})
                db.execute(
                    "UPDATE batches SET finished=?,terminal=?,usable=?,accepted=?,duplicates=?,elapsed=?,tokens=?,response_finish=?,prompt_sha256=? WHERE id=?",
                    (time.time(), term.split(":")[0], len(rows), acc, dup, dur, tok, stop, sha, bid),
                )
            if len(rows) > 0:
                ok_routes += 1
            print(json.dumps({"smoke_route": r.name, "cohort": r.cohort, "terminal": term, "usable": len(rows), "accepted": acc, "elapsed": dur}, ensure_ascii=False), flush=True)
    save_snapshot(db, "all")
    print("SAMPLES_PREVIEW:", json.dumps(samples_preview[:6], ensure_ascii=False, indent=2), flush=True)
    print(f"SMOKE_OK_ROUTES={ok_routes}/{len(ROUTES)}", flush=True)
    return 0 if ok_routes >= 3 else 1


def run_loop(db: sqlite3.Connection, workers: int, batch_limit: int, scope: str) -> bool:
    excl_keys = load_exclusion_keys()
    active_targets = TARGETS if scope == "all" else {scope: TARGETS[scope]}
    routes = [r for r in ROUTES if r.cohort in active_targets]
    with db:
        db.execute("UPDATE batches SET terminal='interrupted', finished=? WHERE terminal IS NULL", (time.time(),))
    counts = totals(db)
    print(json.dumps({"exclusion_keys": len(excl_keys), "unique": counts, "goals": active_targets, "workers": workers}, ensure_ascii=False), flush=True)

    pending = {}
    alive = {r.name: True for r in routes}
    slots = {r.name: r.slots for r in routes}
    targets = {r.name: r.rows for r in routes}
    permanent = set()
    streak = Counter()
    route_active = Counter()
    vendor_active = Counter()
    cohort_active = Counter()
    cooldown = {}
    launches = 0
    last_update = 0.0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while pending or any(counts[c] < n for c, n in active_targets.items()):
            now = time.monotonic()
            active_cohorts = [c for c in active_targets if counts[c] < active_targets[c]]
            while len(pending) < workers and (not batch_limit or launches < batch_limit):
                cands = [
                    r for r in routes
                    if alive[r.name]
                    and counts[r.cohort] < active_targets[r.cohort]
                    and now >= cooldown.get(r.vendor, 0.0)
                    and route_active[r.name] < slots[r.name]
                    and vendor_active[r.vendor] < VENDOR_CAPS.get(r.vendor, 40)
                    and (not cohort_active[r.cohort] or active_targets[r.cohort] - counts[r.cohort] > 80 * cohort_active[r.cohort])
                ]
                if not cands:
                    break
                route = min(
                    cands,
                    key=lambda r: (
                        cohort_active[r.cohort] / max(1, active_targets[r.cohort] - counts[r.cohort]),
                        route_active[r.name] / max(1, slots[r.name]),
                        ROUTES.index(r),
                    ),
                )
                req = min(targets[route.name], max(18, active_targets[route.cohort] - counts[route.cohort]))
                with db:
                    bid = db.execute(
                        "INSERT INTO batches(route,cohort,requested,started,owner) VALUES (?, ?, ?, ?, ?)",
                        (route.name, route.cohort, req, time.time(), scope),
                    ).lastrowid
                fut = executor.submit(batch_result, route, bid, req)
                pending[fut] = (route, bid, req)
                launches += 1
                route_active[route.name] += 1
                vendor_active[route.vendor] += 1
                cohort_active[route.cohort] += 1

            if not pending:
                if not active_cohorts or (batch_limit and launches >= batch_limit):
                    break
                healthy = [r for r in routes if alive[r.name] and r.cohort in active_cohorts]
                if not healthy:
                    break
                next_ready = min(cooldown.get(r.vendor, now) for r in healthy)
                time.sleep(min(max(next_ready - now, 0.2), 15.0))
                continue

            finished, _ = wait(list(pending), timeout=12, return_when=FIRST_COMPLETED)
            for fut in finished:
                route, bid, req = pending.pop(fut)
                route_active[route.name] -= 1
                vendor_active[route.vendor] -= 1
                cohort_active[route.cohort] -= 1
                rows, raw_term, dur, tok, stop, sha = fut.result()
                term = raw_term.split(":")[0]
                acc = dup = 0
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    counts[route.cohort] = db.execute("SELECT COUNT(*) FROM samples WHERE cohort=?", (route.cohort,)).fetchone()[0]
                    for ordinal, text, sc in rows:
                        if counts[route.cohort] + acc >= active_targets[route.cohort]:
                            break
                        v = retain(db, excl_keys, route.cohort, route.name, route.model, text, sc, bid, ordinal, sha)
                        acc += v
                        dup += 1 - v
                    db.execute(
                        "UPDATE batches SET finished=?,terminal=?,usable=?,accepted=?,duplicates=?,elapsed=?,tokens=?,response_finish=?,prompt_sha256=? WHERE id=?",
                        (time.time(), term, len(rows), acc, dup, dur, tok, stop, sha, bid),
                    )
                counts[route.cohort] += acc
                if rows:
                    streak[route.name] = 0
                    if route.name not in permanent:
                        alive[route.name] = True
                    fast = 95 if route.cohort == "deepseek_v41" else 105
                    maximum = 190 if route.cohort == "deepseek_v41" else 160
                    if len(rows) >= 0.78 * req and dur < fast:
                        targets[route.name] = min(targets[route.name] + 10, maximum)
                    if len(rows) < 0.45 * req or dur > 130:
                        targets[route.name] = max(48, int(targets[route.name] * 0.82))
                else:
                    streak[route.name] += 1
                    if term == "rate_limit":
                        slots[route.name] = max(1, slots[route.name] - 1)
                        cooldown[route.vendor] = time.monotonic() + min(90, 24 * streak[route.name])
                        if streak[route.name] >= 8:
                            alive[route.name] = False
                    elif term in {"quota", "auth", "model"}:
                        permanent.add(route.name)
                        alive[route.name] = False
                    elif term in {"no_valid_rows", "temporary", "error"}:
                        targets[route.name] = max(32, int(targets[route.name] * 0.78))
                        if streak[route.name] >= 8:
                            alive[route.name] = False
                    elif streak[route.name] >= 4:
                        alive[route.name] = False
                if launches % 5 == 0 or not rows or acc > 0:
                    print(
                        json.dumps(
                            {"batch": bid, "route": route.name, "term": term, "req": req, "usable": len(rows), "acc": acc, "dup": dup, "elapsed": dur, "cohort_total": counts[route.cohort]},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            if finished and time.monotonic() - last_update >= 10:
                save_snapshot(db, scope)
                last_update = time.monotonic()

    final = save_snapshot(db, scope)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return all(final["unique"][c] >= target for c, target in active_targets.items())


def export_jsonl(db: sqlite3.Connection) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / "bili-v04-synthetic-250k.jsonl.gz"
    staging = out.with_suffix(".writing")
    rows = db.execute(
        "SELECT sample_id, cohort, route, model, text, partition, topic, plugin_mode, length_bucket, scenario_json, batch, ordinal, prompt_sha256 FROM samples ORDER BY rowid"
    )
    count = 0
    with gzip.open(staging, "wt", encoding="utf-8", newline="\n", compresslevel=5) as stream:
        for sid, cohort, route, model, text, part, topic, mode, lb, sc, batch, ordinal, sha in rows:
            payload = {
                "sample_id": sid,
                "text": text,
                "label": "ai",
                "origin_status": "confirmed_ai",
                "split": "train",
                "family": cohort,
                "route": route,
                "model_id": model,
                "partition": part,
                "topic": topic,
                "plugin_mode": mode,
                "length_bucket": lb,
                "batch": batch,
                "ordinal": ordinal,
                "scenario": json.loads(sc),
                "prompt_sha256": sha,
                "source": "zcode_v04_ds" if cohort == "deepseek_v41" else "zcode_v04_other",
            }
            stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    staging.replace(out)
    snap = save_snapshot(db, "all")
    print(json.dumps({"exported": count, "file": str(out), "bytes": out.stat().st_size, "unique": snap["unique"]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--smoke-test", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--export", action="store_true")
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--batch-limit", type=int, default=0)
    parser.add_argument("--cohort", choices=("all", *TARGETS), default="all")
    args = parser.parse_args()
    with closing(connect()) as db:
        if args.status:
            print(json.dumps(report(db), ensure_ascii=False, indent=2))
        elif args.export:
            export_jsonl(db)
        elif args.smoke_test:
            with exclusive_run("all"):
                raise SystemExit(run_smoke(db))
        else:
            with exclusive_run(args.cohort):
                ok = run_loop(db, args.workers, args.batch_limit, args.cohort)
            if not ok and not args.batch_limit:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
