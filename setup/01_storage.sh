#!/usr/bin/env bash
# MetaFind — 建立大型資料儲存區（需要 sudo）
#
# 用法：  bash setup/01_storage.sh
#
# /mnt/data1 目前屬於 klooom，所以要用 sudo 建目錄再 chown 給 kyzen。
# 建完之後 repo 內的 ./data 會 symlink 到這裡，程式一律走 ./data/...，
# 不要在程式裡寫死 /mnt/data1 的絕對路徑。

set -euo pipefail

ROOT=/mnt/data1/kyzen/MetaFind
REPO=/home/kyzen/MetaFindV1
USER_NAME=$(id -un)
GROUP_NAME=$(id -gn)

echo "==> 建立 ${ROOT}"
sudo mkdir -p "${ROOT}"
sudo chown -R "${USER_NAME}:${GROUP_NAME}" /mnt/data1/kyzen

echo "==> 建立子目錄"
mkdir -p "${ROOT}"/sources/{ulip2,objaverse-lvis,procthor}
mkdir -p "${ROOT}"/work                       # shard 暫存，可隨時刪
mkdir -p "${ROOT}"/artifacts/{embeddings,renders,pointclouds,scene_graphs}
mkdir -p "${ROOT}"/artifacts/index/{staging,promoted}
mkdir -p "${ROOT}"/ckpt                       # 內容定址，永不覆寫
mkdir -p "${ROOT}"/runs/{gates,audits,sidecars,progress}
mkdir -p "${ROOT}"/cache/{hf,torch,openclip}  # ViT-bigG-14 約 10GB 會落在這

echo "==> 建立 repo -> 儲存區 symlink"
ln -sfn "${ROOT}" "${REPO}/data"

echo "==> 完成"
df -h /mnt/data1 | tail -1
echo
tree -L 2 "${ROOT}" 2>/dev/null || find "${ROOT}" -maxdepth 2 -type d | sort
echo
echo "驗證：ls -l ${REPO}/data  應指向 ${ROOT}"
ls -l "${REPO}/data"
