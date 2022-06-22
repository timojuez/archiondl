import sys, json, time, traceback, argparse, os, requests, io, string, math, socket, re
from selenium.common.exceptions import StaleElementReferenceException
from threading import Thread, Lock
from splinter import Browser
from splinter.exceptions import ElementDoesNotExist
from PIL import Image, ImageOps
from urllib.parse import urljoin
from requests.exceptions import ConnectionError
from selenium.common.exceptions import TimeoutException, ElementNotInteractableException
from queue import Queue
from concurrent.futures import ThreadPoolExecutor as Executor
from .config import *


dl_file_lock = Lock()
dl_queue = Queue(DOWNLOAD_PROCESSES)


def main():
    #crawl(["https://www.archion.de/de/viewer/?no_cache=1&type=churchRegister&uid=258332"])
    crawl_books()

def crawl_indices():
    with Browser() as browser:
        viewers = CrawlIndices(browser).get_viewers()
    with open("viewers.json", "w") as fp: json.dump(viewers, fp)

def crawl_books():
    with open("viewers.json") as fp: viewers = json.load(fp)
    urls = set([href for _, href in viewers])
    del viewers
    urls_chunked = list(chunks(list(urls), math.ceil(len(urls)/BROWSER_PROCESSES)))
    del urls
    assert(len(urls_chunked) == BROWSER_PROCESSES)
    for urls in urls_chunked: Thread(target=crawl, args=(urls,), daemon=False).start()

def crawl(viewers):
    with Browser('firefox', profile='/home/timo/.mozilla/firefox/9rmc0n9b.archion') as browser:
        BookScraper(browser).scrape_books(viewers)


class Logger:
    _progress = {}

    def update(self, name, progress, total):
        name = str(name)
        if name not in self._progress:
            sys.stderr.write(f"Starting {name}\n")
        if progress == total:
            del self._progress[name]
            sys.stderr.write(f"Finished {name}\n")
        else: self._progress[name] = [progress, total, 0, 1]
        self._update()

    def _update(self):
        sys.stderr.write(" | ".join([f"{p}/{t} ({min(99., 100*p2/t2):02.0f}%)"
            for name, (p, t, p2, t2) in self._progress.items()]))
        sys.stderr.write("\r")
        sys.stderr.flush()

    def set_sub_totals(self, name, total):
        if name not in self._progress: return
        self._progress[name][3] = total

    def notify_download(self, name):
        if name not in self._progress: return
        self._progress[name][2] += 1
        self._update()

logger = Logger()


class AbstractCrawl:

    def __init__(self, b):
        self._b = b

    def login(self):
        try: 
            self._b.visit("https://www.archion.de/de/browse/?no_cache=1")
            self._b.find_by_text("Login").click()
        except (ElementNotInteractableException, ElementDoesNotExist) as e:
            print("Error logging in: %s"%repr(e))
            return
        self._b.find_by_id("user").fill(USER)
        self._b.find_by_id("pass").fill(PASS)
        self.xpath("//*[@name='submit']").click()

    def xpath(self, *args, **xargs):
        return self._b.find_by_xpath(*args, **xargs, wait_time=10)


class CrawlIndices(AbstractCrawl):

    def get_viewers(self):
        self._b.visit("https://www.archion.de/de/browse/?no_cache=1")
        region = self.xpath("//li[contains(text(), 'Braunschweig')]")
        viewers = list(self._get_viewer_urls(region))
        return viewers
        #for e in list(self._get_viewer_urls(region)): self._scrape_book(e)
        #self._scrape_book((["test"], "https://www.archion.de/de/viewer/?no_cache=1&type=churchRegister&uid=277784"))

    def _close_books(self):
        while self._b.is_element_visible_by_xpath("//li[@class='close']"):
            try: self.xpath("//li[@class='close']/a").click()
            except:
                if self._b.is_element_visible_by_xpath("//li[@class='close']"): raise
            finally: time.sleep(.2)

    def _get_viewer_urls(self, region):
        region.click()
        churches = list(self.xpath("//*[@id='mCSB_2']//li[@class='digitallyAvailable']"))
        for i, church in enumerate(churches):
            print(f"Processing church {i}.")
            self._close_books()
            church.click()
            time.sleep(2)
            while True:
                try:
                    books = list(self.xpath("//*[@id='mCSB_5']//li[@class='digitallyAvailable']"))
                    for book in books:
                        try: book.click()
                        except:
                            print(f"Error in book.click(). books = {books}")
                            raise
                        link = self.xpath("//*[contains(@href, 'viewer')]")["href"]
                        path = [self.xpath(f"//*[@class='lvl-{i}']").text.strip() for i in (1,2,3,4)]
                        path = list(filter(lambda e:e, path))
                        #print(path[1:], link)
                        r = (path, link)
                        with open("viewers","a") as fp: fp.write("%s\n"%json.dumps(r))
                        yield r
                except StaleElementReferenceException: pass
                else: break


class BookScraper(AbstractCrawl):
    login_lock = Lock()
    cookies = None

    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        if os.path.exists("downloaded"):
            with open("downloaded") as fp: self.downloaded = [e.strip() for e in fp]
        else: self.downloaded = []
        self._applied_cookies = None
        self.login()

    def login(self):
        """ thread safe login """
        with self.login_lock:
            if self.cookies == self._applied_cookies:
                # login and store cookies
                super().login()
                self.__class__.cookies = self._b.cookies.all()
            else:
                # apply cookies
                self._b.visit("https://www.archion.de/")
                self._b.cookies.delete_all()
                self._b.cookies.add(self.cookies)
                self._applied_cookies = self.cookies

    def scrape_books(self, urls):
        for url in urls:
            while True:
                try: self.scrape_book(url)
                except (Exception, TimeoutException, ConnectionError, socket.gaierror, ElementDoesNotExist):
                    traceback.print_exc()
                    time.sleep(10)
                    self.login()
                else: break

    def scrape_book(self, href, pages=None):
        if href in self.downloaded or pages:
            #print(f"Already downloaded {href}")
            return
        self._b.visit(href)
        path = [e.text for e in list(self.xpath("//*[@class='dvbreadcrumb']/a"))[1:]]
        for i in range(ZOOM):
            self.xpath("//a[@class='zoom-in']").click()
            time.sleep(1)
        pages_e = list(self.xpath("//select[@class='page-select']/option"))
        for i, page in enumerate(pages_e):
            if pages and i not in pages: continue
            digits = len(pages_e[-1]["value"])
            filename = self._make_filename(path, page["value"].zfill(digits)+".jpg")
            if os.path.exists(filename): continue
            logger.update(path, i+1, len(pages_e))
            page.click()
            tiles = [tile["_src"] for tile in self.xpath("//*[@class='zoom-tiles']/img")]
            #time.sleep(1)
            #tiles = [s.replace("&amp;","&") for s in re.findall('_src="([^"]*)"', self.xpath("//*[@class='zoom-tiles']").html)]
            im = self._unite_tiles(str(path), [(*map(int, re.findall("/(\d*)_(\d*)\.jpg", src)[0]), urljoin(self._b.url, src))
                for src in tiles])
            im.save(filename,optimize=True,quality=JPEG_QUALITY)
        with dl_file_lock:
            with open("downloaded", "a") as fp: fp.write(f"{href}\n")

    def _download(self, name, tiles, downloaded):
        for i, (x,y,url) in enumerate(tiles):
            for j in range(3):
                try: 
                    resp = requests.get(url, timeout=4)
                    if resp.status_code != 200: raise Exception(f"Return code {resp.status_code} != 200")
                except Exception as e:
                    sys.stderr.write(f"Exception downloading url {url}: {repr(e)}\n")
                    if j == 2:
                        #raise
                        return
                    time.sleep(1)
                else: break
            logger.notify_download(name)
            try: im = Image.open(io.BytesIO(resp.content))
            except:
                print(f"Exception opening image {url}")
                raise
            downloaded.append((x,y,im))

    def _unite_tiles(self, name, tiles):
        downloaded = []
        threads = [
            Thread(target=self._download, args=(name, tiles_, downloaded), daemon=True, name=f"Download {name}")
            for tiles_ in chunks(tiles, math.ceil(len(tiles)/DOWNLOAD_PROCESSES))]
        logger.set_sub_totals(name, len(tiles))
        #print(f"\nDownloading {len(tiles)} tiles in {len(threads)} threads.")
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        if not len(downloaded) == len(tiles): raise ConnectionError("Download unsuccessful.")
        #for i, (x, y, url) in enumerate(tiles):
        #    downloaded.append((x,y,Image.open(io.BytesIO(requests.get(url).content))))
        #    #sys.stderr.write(f"\rDownloading ({i+1}/{len(tiles)})")
        #    #sys.stderr.flush()
        #tiles = [(x,y,Image.open(io.BytesIO(requests.get(url).content))) for x, y, url in tiles]
        tiles = downloaded
        #sys.stderr.write("\n")
        #sys.stderr.flush()
        total_width = sum([im.size[0] for x,y,im in tiles if y==0])
        total_height = sum([im.size[1] for x,y,im in tiles if x==0])
        #width, height = tiles[0][2].size
        widths = [tile[2].size[0] for tile in tiles]
        heights = [tile[2].size[1] for tile in tiles]
        width = max(widths, key = widths.count)
        height = max(heights, key = heights.count)
        new = Image.new('RGB' if COLOUR else 'L',(total_width, total_height))
        for x,y,im in tiles: new.paste(im, (x*width, y*height))
        return new

    def _make_filename(self, path, filename):
        path = [sanitize(p) for p in path]
        filename = sanitize(filename)
        for i in range(len(path)):
            try: os.mkdir(os.path.join(*list(path[:(i+1)])))
            except FileExistsError: pass
        return os.path.join(*path, filename)


def sanitize(s):
    valid_chars = "ÄÖÜäöüß-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in s if c in valid_chars)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__': main()

