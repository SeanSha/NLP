import os
import gdown
import zipfile

class CategorizedCorpus:
    def __init__(self, path):
        self.path = path
        self.folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    def __iter__(self):

        for category in self.folders:

            dir_path = os.path.join(self.path, category)

            for file in os.listdir(dir_path):

                file_path = os.path.join(dir_path, file)

                if os.path.isfile(file_path):

                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read().strip()
                        yield text, category
        

if __name__ == "__main__":

    cloud_storage_url = "https://www.dropbox.com/scl/fo/12vifl746lgph6skaiimx/AC3Dhh5k0CzGWPNB4k8_SwQ?rlkey=2z9u8zls7nax2oujgyve5p3ay&st=5tmwa9cq&dl=0"

    # 自定义下载路径和解压路径
    output_path = "/Users/sean/Projects/Data/download.zip"  # 下载文件的存储路径
    extract_path = "/Users/sean/Projects/Data/NLP"  # 解压后的文件存储路径

    # 下载文件
    gdown.download(cloud_storage_url, output_path, quiet=False)

    # 解压文件到指定路径
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # 创建 CategorizedCorpus 实例，传入解压后的路径
    CC = CategorizedCorpus(extract_path)

    for text, category in CC:
        print(f"{text[:60]}...{category}")