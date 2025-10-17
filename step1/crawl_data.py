import os
import time
import zipfile
from xml.etree import ElementTree as ET
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

DOCX_FILE = r"dataset\Danh_muc_Bo_luat_va_Luat_cua_Viet_Nam_2909161046 (1).docx"
DOWNLOAD_DIR = os.path.abspath("dataset/downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def get_hyperlinks_from_docx(docx_path):
    links = []
    with zipfile.ZipFile(docx_path, 'r') as z:
        doc_xml = z.read("word/document.xml").decode("utf-8")
        try:
            rels_xml = z.read("word/_rels/document.xml.rels").decode("utf-8")
        except KeyError:
            rels_xml = None

    rels = {}
    if rels_xml:
        root = ET.fromstring(rels_xml)
        for rel in root:
            rId = rel.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}Id") or rel.attrib.get("Id")
            target = rel.attrib.get("Target")
            if rId and target:
                rels[rId] = target

    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
          'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    doc_root = ET.fromstring(doc_xml)
    for hyp in doc_root.findall('.//w:hyperlink', ns):
        rid = hyp.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        texts = [t.text for t in hyp.findall('.//w:t', ns) if t.text]
        display = ''.join(texts)
        url = rels.get(rid)
        if url and "luatvietnam.vn" in url: 
            links.append((display, url))
    return links

def start_driver(download_dir):
    chrome_options = Options()
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def try_click_vietnamese_links(driver):
    anchors = driver.find_elements(By.CSS_SELECTOR, "div.download-vb a[href]")
    clicked_any = False
    for a in anchors:
        href = a.get_attribute("href")
        if not href or href.startswith("javascript:"):
            continue
        if any(ext in href.lower() for ext in [".doc", ".docx"]):
            try:
                driver.execute_script("window.open(arguments[0], '_blank');", href)
                clicked_any = True
                time.sleep(0.5)
            except:
                pass
    return clicked_any

def wait_for_active_downloads(download_folder, timeout=120):
    t0 = time.time()
    while True:
        files = os.listdir(download_folder)
        in_progress = [f for f in files if f.endswith(".crdownload") or f.endswith(".part")]
        if not in_progress:
            return
        if time.time() - t0 > timeout:
            print("Timeout chờ download hoàn tất.")
            return
        time.sleep(1)

def visit_and_download_links(driver, links):
    for idx, (display, url) in enumerate(links, 1):
        print(f"[{idx}/{len(links)}] Visiting: {display} -> {url}")
        try:
            driver.get(url)
            time.sleep(1)
            clicked = try_click_vietnamese_links(driver)
            if clicked:
                print("Đã click link VB tiếng Việt, chờ download...")
                wait_for_active_downloads(DOWNLOAD_DIR)
        except Exception as e:
            print("Error xử lý link:", e)

def main():
    links = get_hyperlinks_from_docx(DOCX_FILE)
    print(f"Tìm thấy {len(links)} hyperlink.")

    driver = start_driver(DOWNLOAD_DIR)

    print("Mở Chrome xong. Vui lòng đăng nhập vào trang web nếu cần.")
    input("Sau khi đăng nhập xong, nhấn Enter để tiếp tục tải file...")

    try:
        visit_and_download_links(driver, links)
    finally:
        print("Hoàn tất, giữ Chrome mở hoặc tắt tùy bạn.")
        driver.quit()

if __name__ == "__main__":
    main()
