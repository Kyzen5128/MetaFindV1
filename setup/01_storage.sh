#!/usr/bin/env bash
# MetaFind — 建立資料儲存區
#
# 用法：
#   bash setup/01_storage.sh                                   # 資料放在 <repo>/data/
#   METAFIND_DATA=/mnt/big/MetaFind bash setup/01_storage.sh    # 放到別的磁碟
#
# 目錄結構由 metafind/paths.py 決定，這裡不重複宣告。上一版自己列了一組
# sources/ artifacts/ ckpt/ runs/ 的目錄，而程式碼實際用的是
# datasets/ models/ outputs/ —— 兩邊從來沒對過，跑完 setup 會得到一棵半空的樹，
# 外加一堆沒有任何程式讀寫的目錄。
#
# 也不再 sudo。上一版寫死前一台機器上另一個使用者的磁碟路徑並 chown 整個目錄；
# 在任何其他 checkout 上，它要嘛失敗，要嘛動到不該動的東西。要放到大容量磁碟
# 就設 METAFIND_DATA，權限由使用者自己準備。

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${METAFIND_PYTHON:-python3}"
cd "$REPO"

eval "$("$PY" -m metafind.paths)"

echo "==> 資料根目錄 ${METAFIND_DATA}"
mkdir -p "$METAFIND_DATA"

# 只有在 METAFIND_DATA 指向 repo 外面時才需要 symlink。
if [ "$(cd "$METAFIND_DATA" && pwd -P)" != "$(cd "$REPO/data" 2>/dev/null && pwd -P || echo none)" ]; then
  ln -sfn "$METAFIND_DATA" "$REPO/data"
  echo "==> ${REPO}/data -> ${METAFIND_DATA}"
fi

echo "==> 建立目錄（清單來自 paths.ALL_OUTPUT_DIRS，不在此重列）"
"$PY" -c "from metafind import paths; paths.ensure_dirs()"
mkdir -p "$METAFIND_DATASETS" "$METAFIND_MODELS" "$METAFIND_HF_CACHE"

echo "==> 完成"
df -h "$METAFIND_DATA" | tail -1
echo
"$PY" -c "from metafind import paths; print(paths.describe())"
