# FPGA 工业缺陷检测

FPGA 工业缺陷检测相关文档、可执行文件、代码、模型与交互网页的汇总仓库。

较大文件（如 FPGA 与树莓派的系统镜像）体积超出 GitHub 限制，已放在网盘，下载地址见
`FPGA&rpi img/文件提取.txt`。

> **核心规则**：模型与标签配合使用，**一个模型对应一个 LABEL**，两者标签必须一一对应。
> LABEL 文件**无后缀**，且换行编码**必须为 `\n`（LF）**，使用 `\r\n`（CRLF）会导致模型无法正常识别。

---

## 一、仓库文件结构（当前已推送 497 个文件 / 约 234 MB）

```
FPGA-INDUSTRY/
├─ arm/                                  机械臂控制源码（2 个文件）
│  ├─ fast.txt
│  └─ oringenal.txt
├─ document/                             项目与平台文档（5 份 PDF）
│  ├─ 智能产线应用赛说明(1).pdf
│  ├─ paddlepaddle模型训练指导手册V1.4.pdf
│  ├─ paddlepaddle模型训练指导手册V1.6(1).pdf
│  ├─ 工业缺陷检测实训平台开发指导书V4.0_01.pdf
│  └─ 工业缺陷检测实训平台（创新版）用户手册V1.0_04.pdf
├─ edge(red)/                            边缘端缺陷检测（421 个文件）
│  ├─ *.py                               20 个 OpenCV 检测脚本（迭代版本）
│  ├─ 0010.txt                           一份脚本的文本备份
│  ├─ input_images/                      原始样本 200 张（4 类 × 50）
│  │  ├─ ca_shang*.jpg                   擦伤
│  │  ├─ zang_wu*.jpg                    脏污
│  │  ├─ zhe_zhou*.jpg                   褶皱
│  │  └─ zhen_kong*.jpg                  针孔
│  └─ output_images/                     检测结果 200 张（与 input 一一同名）
├─ FPGA&rpi img/                         FPGA / 树莓派系统制作（35 个文件）
│  ├─ 20250627系统烧录步骤.doc
│  ├─ 文件提取.txt                        网盘地址与提取码
│  ├─ 系统卡制作步骤/                      烧录步骤文档 + 烧录工具
│  ├─ 系统格式化工具/                      DiskGenius 及配套 DLL / 日志
│  └─ 系统烧录工具/                        win32diskimager 安装包
├─ modle&lable/                          模型与标签（22 个文件，见第二节）
├─ v1 WEB/                               论文模板（1 份 txt + 4 份 html）
├─ v2 WEB/                               DeepSeek 缺陷分析网页
│  ├─ start.bat                          双击启动
│  ├─ server.py                          本地代理服务（转发 DeepSeek，零依赖）
│  ├─ index.html                         页面（图表 + 多轮对话）
│  ├─ config.json                        接口配置
│  └─ 使用说明.md                         完整使用文档
├─ 秘籍PROMAXPLUS.txt                     常用运维命令速查（服务启动/调速）
└─ .gitignore                            排除 Python 缓存与系统杂项
```

各目录说明：

| 目录 | 用途 | 关键内容 |
| --- | --- | --- |
| `arm/` | 机械臂控制 | 基于 Flask + 串口的抓取控制源码（`Arm_Lib`、气泵继电器指令） |
| `document/` | 文档资料 | 平台开发指导书、用户手册、PaddlePaddle 训练手册、赛事说明 |
| `edge(red)/` | 边缘端检测 | 传统 CV 脚本（自适应阈值 + 轮廓提取）与 400 张缺陷样本 |
| `FPGA&rpi img/` | 部署环境 | 系统卡制作、烧录、格式化的工具与步骤文档 |
| `modle&lable/` | 模型与标签 | 多个模型版本，每个版本配同名目录下的 `label_list` |
| `v1 WEB/` | 论文材料 | 论文排版模板 |
| `v2 WEB/` | 交互网页 | 支持 DeepSeek 云端的缺陷数据分析平台 |
| `秘籍PROMAXPLUS.txt` | 运维速查 | `systemctl` 启停检测服务、机械臂服务、调速 |

---

## 二、模型与标签对应关系

**一个模型（`.nb`）必须与同目录下的 `label_list` 配套使用**，两者顺序必须完全对应，不能混用其他版本的标签文件。

| 模型目录 | 模型文件 | 标签文件 | 类别数 | 标签顺序 |
| --- | --- | --- | --- | --- |
| `2(PM)` | `ssd_mobilenet_v1_opt.nb`（另有 `.nb.backup`） | `label_list` | 6 | zhen_kong, ca_shang, zang_wu, zhe_zhou, zheng_chang, liebian |
| `Final mode` | `ssd_mobilenet_v1_opt.nb` | `label_list` | 6 | 同上 |
| `BU` | `ssd_mobilenet_v1_opt.nb` | `label_list0` / `label_list123.txt` ⚠ 无无后缀版本 | 6 | ca_shang, lie_bian, zang_wu, zhe_zhou, zhen_kong, zheng_chang |
| `BU1/模型-5种缺陷` | `ssd_mobilenet_v1_opt.nb` | `label_list` | 6 | 同 `BU`（5 类缺陷 + 正常） |
| `BU2/模型-4种缺陷` | `ssd_mobilenet_v1_opt.nb` | `label_list` | 5 | ca_shang, zang_wu, zhe_zhou, zhen_kong, zheng_chang |
| `新机器模型` | `ssd_mobilenet_v1_opt.nb` | `label_list`（另有 `.bchjc` / `.qxjc` 变体） | 3 | zang_wu, zhe_zhou, zheng_chang |
| `集创赛` | `ssd_mobilenet_v1_opt.nb` | `label_list` | 5 | ca_shang, zang_wu, zhe_zhou, zhen_chang ⚠, zhen_kong |
| `全无裂边90` | `ssd_mobilenet_v1_opt.nb` | 无 | — | 需自行配 `label_list` |
| `最初版` | `ssd_mobilenet_v1_opt.nb` | 无 | — | 需自行配 `label_list` |

标签中文对照：

| 标签 | 含义 |
| --- | --- |
| `ca_shang` | 擦伤 |
| `zang_wu` | 脏污 |
| `zhe_zhou` | 褶皱 |
| `zhen_kong` | 针孔 |
| `liebian` / `lie_bian` | 裂边 |
| `zheng_chang` | 正常 |

> `新机器模型/run.sh` 与 `exclude.list` 为板端部署脚本，模型部署路径为 `/opt/ssd_detection_demo`，
> 标签需放在 `/opt/ssd_detection_demo/labels/label_list`。

---

## 三、三条硬性规则

### 1. 一个模型对应一个 LABEL

每个 `.nb` 模型只认与它配套训练的那份 `label_list`。换模型必须同时换标签文件，且**标签顺序不能调整**。

### 2. LABEL 文件无后缀

板端读取的是不带扩展名的 `label_list`。带 `.txt` / `.bchjc` / `.qxjc` 后缀的文件不能直接用于部署，
需改名为 `label_list`。

### 3. 换行必须是 LF（`\n`）

用 Windows 记事本编辑过的标签文件会变成 CRLF（`\r\n`），板端解析时末尾的 `\r` 会被当成标签名的一部分，
导致类别匹配失败、识别结果为空或全部判错。

检查并修复：

```powershell
# 检查是否有 CRLF
$b = [System.IO.File]::ReadAllBytes("label_list")
[System.Text.Encoding]::UTF8.GetString($b).Contains("`r`n")

# 修复：把 CRLF 全部换成 LF 后写回
$t = [System.IO.File]::ReadAllText("label_list") -replace "`r`n", "`n"
[System.IO.File]::WriteAllText("label_list", $t, (New-Object System.Text.UTF8Encoding($false)))
```

Linux / 板端可用：

```bash
sed -i 's/\r$//' label_list
```

---

## 四、已知问题（待修正）

1. **`新机器模型/label_list` 是 CRLF 换行**，直接部署会导致识别异常，需按第三节第 3 条转成 LF。
2. **标签名拼写不一致**：`BU`、`BU1` 用 `lie_bian`，而 `2(PM)`、`Final mode` 用 `liebian`；
   `集创赛/label_list` 里写的是 `zhen_chang`，其余版本是 `zheng_chang`。
   这类不一致会让不同模型输出的类别名对不上，统计与显示前需统一。
3. **`BU` 缺少无后缀的 `label_list`**，只有 `label_list0` 和 `label_list123.txt`，部署前需补一份。
4. **`全无裂边90`、`最初版` 只有模型没有标签**，单独使用时需自行配套 `label_list`。
5. 仓库为公开仓库，`FPGA&rpi img/` 下的 `DiskGenius.exe`、`win32diskimager`、`DGFileViewer.exe`
   属第三方商业软件，公开分发可能涉及授权问题，按需决定是否移出仓库。

---

## 五、本地更新与推送

```powershell
cd "D:\github test1"

# 首次配置（已配置过可跳过）
git config user.name  "15010120896chy-ai"
git config user.email "15010120896chy-ai@users.noreply.github.com"

# 国内网络直连 GitHub 会连接被重置，需走本地代理
git config http.proxy  http://127.0.0.1:7890
git config https.proxy http://127.0.0.1:7890

# 日常提交与推送
git add -A
git commit -m "更新说明"
git push
```

> 单文件上限 100 MB，超过会被拒绝；更大的数据集请放网盘并在本文档登记。
