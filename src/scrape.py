import sys, json, time, traceback, argparse, os, requests, io, string, math, socket, re, cv2
from selenium.common.exceptions import StaleElementReferenceException
from threading import Thread, Lock, Semaphore
from splinter import Browser
from splinter.exceptions import ElementDoesNotExist
from urllib.parse import urljoin
from requests.exceptions import ConnectionError
from selenium.common.exceptions import TimeoutException, ElementNotInteractableException
import numpy as np
from concurrent.futures import ThreadPoolExecutor as Executor
import concurrent.futures
from .config import *


no_parallel_execution_lock = Lock()
def no_parallel_execution(func):
    def f(*args, **xargs):
        with no_parallel_execution_lock: return func(*args, **xargs)
    return f

def retry_after_exception(func):
    def f(*args, **xargs):
        while True:
            try: return func(*args, **xargs)
            except Exception:
                traceback.print_exc()
                time.sleep(10)
    return f

def sanitize(s):
    valid_chars = "ÄÖÜäöüß-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in s if c in valid_chars)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    #crawl(["https://www.archion.de/de/viewer/?no_cache=1&type=churchRegister&uid=258332"])
    crawl_books()

def crawl_indices():
    with Browser() as browser:
        viewers = CrawlIndices(browser).get_viewers()
    if not DRYRUN:
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
    _buffer = (0, 1)

    def update_buffer(self, semaphore):
        self._buffer = [semaphore.used, semaphore.total]
        self._update()

    def starting_page(self, filename, page_no, page_total, tiles_total):
        self._progress[filename] = [page_no, page_total, 0, tiles_total]
        self._update()

    def finished_page(self, filename):
        del self._progress[filename]
        self._update()

    def notify_download(self, filename, *args):
        if filename not in self._progress: return
        self._progress[filename][2] += 1
        self._update()

    def _update(self):
        b_used, b_total = self._buffer
        sys.stderr.write("\033[K")
        sys.stderr.write(f"Buffer: {b_used:0{len(str(b_total))}d}/{b_total} | ")
        sys.stderr.write(" | ".join([f"{p}/{t} ({min(99., 100*p2/t2):02.0f}%)"
            for name, (p, t, p2, t2) in self._progress.items()]))
        sys.stderr.write("\r")
        sys.stderr.flush()

logger = Logger()


class AbstractCrawl:

    def __init__(self, b):
        self._b = b

    @no_parallel_execution
    @retry_after_exception
    def login(self):
        self._b.visit("https://www.archion.de/de/browse/?no_cache=1")
        try: self._b.find_by_text("Login").click()
        except (ElementNotInteractableException, ElementDoesNotExist) as e:
            print("Error logging in: %s"%repr(e))
            return
        self._b.find_by_id("user").fill(USER)
        self._b.find_by_id("pass").fill(PASS)
        self.xpath("//*[@name='submit']").click()

    def xpath(self, *args, **xargs):
        return self._b.find_by_xpath(*args, **xargs, wait_time=10)


class CrawlIndices(AbstractCrawl):

    @no_parallel_execution
    def get_viewers(self):
        self._b.visit("https://www.archion.de/de/browse/?no_cache=1")
        region = self.xpath("//li[contains(text(), 'Braunschweig')]")
        viewers = list(self._get_viewer_urls(region))
        return viewers
        #for e in list(self._get_viewer_urls(region)): self._scrape_book(e)

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


class CountingSemaphore(Semaphore):

    def __init__(self, value):
        super().__init__(value)
        self.total = value
        self.used = 0

    def acquire(self, *args, **xargs):
        r = super().acquire(*args, **xargs)
        self.used += 1
        logger.update_buffer(self)
        return r

    def release(self, *args, **xargs):
        r = super().release(*args, **xargs)
        self.used -= 1
        logger.update_buffer(self)
        return r


class BookScraper(AbstractCrawl):
    login_lock = Lock()
    cookies = None
    _url_crawler_semaphore = CountingSemaphore(URL_BUFFER_SIZE)
    _executor = Executor(max_workers=DOWNLOAD_PROCESSES)

    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        if os.path.exists("downloaded"):
            with open("downloaded") as fp: self.downloaded = [e.strip() for e in fp]
        else: self.downloaded = []
        self.login()

    def scrape_books(self, urls, *args, **xargs):
        for url in urls:
            self.scrape_book(url, *args, **xargs)

    def scrape_book(self, *args, **xargs):
        while True:
            try: self._scrape_book(*args, **xargs)
            except (Exception, TimeoutException, ConnectionError, socket.gaierror, ElementDoesNotExist) as e:
                if isinstance(e, KeyboardInterrupt): raise
                traceback.print_exc()
                time.sleep(10)
                self.login()
            else: break

    @no_parallel_execution
    def _scrape_book(self, href, pages=None):
        """ pages: list of int, pages that should be downloaded """
        if href in self.downloaded and pages == None:
            #print(f"Already downloaded {href}")
            return
        def on_finish_book():
            if not DRYRUN:
                with open("downloaded", "a") as fp: fp.write(f"{href}\n")
            print(f"Finished {path}")

        self._b.visit(href)
        path = [e.text for e in list(self.xpath("//*[@class='dvbreadcrumb']/a"))[1:]]
        print(f"Starting {path}")
        for i in range(ZOOM):
            self.xpath("//a[@class='zoom-in']").click()
            time.sleep(1)
        pages_e = list(self.xpath("//select[@class='page-select']/option"))
        finished_pages = set()
        def mark_page_done(page_no):
            finished_pages.add(page_no)
            if len(finished_pages) == len(pages_e): on_finish_book()
        for page in pages_e:
            page_no = int(page["value"]) #FIXME +1
            if pages and page_no not in pages: continue
            digits = len(pages_e[-1]["value"])
            filename = self._make_filename(path, str(page_no).zfill(digits)+".jpg")
            if os.path.exists(filename) and not DRYRUN:
                mark_page_done(page_no)
                continue
            def on_finish_page(page_no_=page_no, filename_=filename):
                logger.finished_page(filename_)
                mark_page_done(page_no_)
            page.click()
            for _ in range(3):
                tiles_src = [tile["_src"] for tile in self.xpath("//*[@class='zoom-tiles']/img")]
                if tiles_src: break
                else: time.sleep(3)
            if len(tiles_src) == 0: raise Exception(f"No tiles found on {href}, p. {page_no}.")
            logger.starting_page(filename, page_no, len(pages_e), len(tiles_src))
            futures = []
            positions = []
            for src in tiles_src:
                tile_url = urljoin(self._b.url, src)
                x, y = tuple(map(int, re.findall("/(\d*)_(\d*)\.jpg", src)[0]))
                self._url_crawler_semaphore.acquire()
                future = self._executor.submit(self._download, filename, tile_url, page_no)
                futures.append(future)
                positions.append((x,y))
            Thread(target=self._unite_tiles, args=(filename, (href, pages), positions, futures, on_finish_page),
                name=f"Unite_tiles: {filename}").start()

    def _download(self, filename, url, page_no):
        try:
            for j in range(3):
                try:
                    resp = requests.get(url, timeout=4)
                    if resp.status_code != 200: raise Exception(f"Return code {resp.status_code} != 200")
                except Exception as e: # ReadTimeout
                    if j == 2:
                        sys.stderr.write(f"Exception downloading url {url} to {filename}.\n")
                        raise
                    time.sleep(1)
                else:
                    logger.notify_download(filename, page_no)
                    return cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        finally:
            self._url_crawler_semaphore.release()

    def _unite_tiles(self, filename, scrape_book_args, positions, futures, on_success=None):
        """
            on_success: callback that will be called after this method has finished
        """
        done, not_done = concurrent.futures.wait(futures, timeout=None,
            return_when=concurrent.futures.FIRST_EXCEPTION)
        try: tiles = [(x, y, future.result()) for (x, y), future in zip(positions, futures)]
        except:
            # Exception in _download()
            traceback.print_exc()
            for f in not_done:
                if f.cancel(): self._url_crawler_semaphore.release()
            self._scrape_book(*scrape_book_args) # retry # TODO: only once
            return
        total_width = sum([im.shape[1] for x,y,im in tiles if y==0])
        total_height = sum([im.shape[0] for x,y,im in tiles if x==0])
        widths = [tile[2].shape[1] for tile in tiles]
        heights = [tile[2].shape[0] for tile in tiles]
        #width = max(widths, key = widths.count)
        #height = max(heights, key = heights.count)
        im = np.zeros((total_height, total_width, 3), np.uint8)
        tiles = sorted(tiles, key=lambda e:(e[1], e[0]))
        x_ = 0
        y_ = 0
        heights = []
        for x, y, tile in tiles:
            if x==0:
                x_=0
                if heights: y_ += max(heights, key=heights.count)
                heights = []
            heights.append(tile.shape[0])
            im[y_:y_+tile.shape[0], x_:x_+tile.shape[1]] = tile
            x_ += tile.shape[1]
        #for x, y, tile in tiles: im[y*height:y*height+tile.shape[0], x*width:x*width+tile.shape[1], :3] = tile
        #max_x = max([x for x, y, tile in tiles])+1
        #np.concatenate([np.concatenate([tile for x, y, tile in tiles if x == x_], axis=1) for x_ in range(max_x)]
        if not DRYRUN: self._saveimg(filename, im)
        if DEBUG_OUTPUT: self._saveimg("debug.jpg", im)
        if on_success: on_success(filename)

    def _saveimg(self, filename, im):
        cv2.imwrite(filename, im, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

    def _make_filename(self, path, filename):
        path = [sanitize(p) for p in path]
        filename = sanitize(filename)
        for i in range(len(path)):
            try: os.mkdir(os.path.join(*list(path[:(i+1)])))
            except FileExistsError: pass
        return os.path.join(*path, filename)


if __name__ == '__main__': main()

