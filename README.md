# d3ft-train — d3-flash 训练环境云端打包（GitHub Actions）

利用 GitHub Actions 在云端打训练环境镜像（GitHub runner = 有外网的 Linux x86_64），
产物自动挂到本仓库 Release，浏览器即可下载，传到无外网服务器使用。

**底座模型不打进镜像**：模型手动下载，运行时用 `-v` 挂载。

## 使用

1. 本仓库已包含全部内容（工作流 + `d3-kit/` + 手册），直接推送到 GitHub 即可
   （`.gitignore` 已排除超过 100M 限制的 `d3-kit/runtime/*.tar.gz`）
2. GitHub → **Actions** → **build-offline** → **Run workflow**
   - 镜像内置 llama.cpp b11588（官方 CUDA 12.8 预编译包，与生产同版本）
   - `include_wheels` 不用勾（裸机路线A 离线安装才需要）
3. 运行完成后到 **Releases** 下载全部资产：
   `d3ft-train-image.tar.gz.part-00/01/...`（镜像分卷）、`*.sha256`（校验）、
   `Miniconda3-py311-*.sh`（裸机路线A才用）

## 服务器端还原镜像

```bash
cat d3ft-train-image.tar.gz.part-* > d3ft-train-image.tar.gz
sha256sum -c d3ft-train-image.tar.gz.sha256
docker load -i d3ft-train-image.tar.gz
```

## 模型手动下载（任意有网机器）

```bash
pip install -U modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
modelscope download --model vllm-sr/d3-flash --local_dir d3-flash
# 把 d3-flash 整目录上传到服务器: ~/d3-work/models/d3-flash/
```

## 服务器运行（CentOS 7 / 无外网 GPU 服务器，路线B）

```bash
# 1) 起容器
docker run --gpus all --rm -it --shm-size 8g \
  -v ~/d3-work:/root/d3-work -v ~/d3-kit:/root/d3-kit \
  -e D3_WORK=/root/d3-work -e D3_PY=python3 d3ft-train:latest bash

# 2) 容器内初始化（识别镜像内置 llama.cpp / 转换工具链 / 训练补丁 / 终检；模型已挂载则自动跳过拷贝）
bash /root/d3-kit/install_offline.sh
bash /root/d3-kit/check_env.sh        # 网络 FAIL 属离线预期，忽略
```

完整流程（打包 → 下载还原 → 模型摆放 → 训练/导出/验证）见本目录
**《Docker训练手册.md》**；背景与路线对比见上级《决策模型部署训练手册-新服务器版.md》第 5 节。
