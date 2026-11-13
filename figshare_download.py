import os
import requests
import hashlib
from tqdm import tqdm

# ===================== 配置区，你改这里 =====================
SAVE_ROOT = r"F:\new\data"  # 目标文件夹
article_id_list = [
    28931402,   # ADI
    28936250,   # LYM
    28939016,   # DEB
    28939100,   # MUC
    28939115,   # MUS
    28939151,   # NORM
    28939169,   # STR
    28939460    # TUM
]
# 你的Clash混合端口是7897
# proxies = None
proxies = {
    "http": "http://127.0.0.1:7897",
    "https": "http://127.0.0.1:7897"
}
# ==========================================================

os.makedirs(SAVE_ROOT, exist_ok=True)

def md5_check(filepath, expect_md5):
    m = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024*1024):
            m.update(chunk)
    return m.hexdigest() == expect_md5

def download_with_resume(url, save_path, proxies):
    headers = {}
    if os.path.exists(save_path):
        exist_size = os.path.getsize(save_path)
        headers["Range"] = f"bytes={exist_size}-"
    else:
        exist_size = 0

    resp = requests.get(url, headers=headers, stream=True, proxies=proxies, timeout=60)
    if resp.status_code not in (200, 206):
        print(f"❌ 请求失败，code={resp.status_code}")
        return False

    total_size = int(resp.headers.get("content-length", 0)) + exist_size
    mode = "ab" if exist_size > 0 else "wb"

    with open(save_path, mode) as f, tqdm(
        desc=os.path.basename(save_path),
        initial=exist_size,
        total=total_size,
        unit="B",
        unit_scale=True
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192*4):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    return True

for aid in article_id_list:
    print(f"\n==== 正在拉取 article {aid} ====")
    try:
        res = requests.get(f"https://api.figshare.com/v2/articles/{aid}/files", proxies=proxies, timeout=30)
        res.raise_for_status()
        file_list = res.json()
        for f in file_list:
            dl_url = f"https://figshare.com/ndownloader/files/{f['id']}"
            fname = f["name"]
            fmd5 = f["md5"]
            save_file = os.path.join(SAVE_ROOT, fname)

            if os.path.exists(save_file):
                if md5_check(save_file, fmd5):
                    print(f"✅ {fname} 已经下载完好，跳过")
                    continue
                else:
                    print(f"⚠️ {fname} MD5不对，重新下载")
                    os.remove(save_file)

            print(f"开始下载：{fname}")
            ok = download_with_resume(dl_url, save_file, proxies)
            if ok and md5_check(save_file, fmd5):
                print(f"✅ {fname} 下载+校验成功")
            else:
                print(f"❌ {fname} 校验失败")
    except Exception as e:
        print(f"获取article {aid} 出错：{e}")
