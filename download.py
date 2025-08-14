import requests


for i in range(0, 144):
    file_num = str(i).zfill(3)
    filename = f"demowki{file_num}.jpg"
    url = f"https://klient.fotoklaser.pl/download.php?mode=api_preview&access=oGywJNAeoELTy4k_2_KE&file={filename}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"data/{filename}", "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename} (status code: {response.status_code})")

    print(f"Downloaded {i} files")