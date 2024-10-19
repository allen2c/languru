import asyncio
import json
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, List, Literal, Optional, Sequence, Text, Union

from diskcache import Cache
from playwright.async_api import BrowserContext, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from yarl import URL

from languru.config import console
from languru.types.web.documents import HtmlDocument
from languru.utils._playwright import (
    get_page,
    handle_captcha_page,
    simulate_human_behavior,
    try_close_page,
)
from languru.utils.common import debug_print_banner, try_await_or_none
from languru.utils.crawler import (
    add_extra_params_to_url,
    escape_query,
    filter_out_extensions,
)
from languru.utils.html_parser import (
    as_markdown,
    drop_all_comments,
    drop_all_tags,
    drop_no_used_attrs,
)
from languru.web.remote import get_search_engines

cache = Cache(Path.home().joinpath(".languru/data/cache/web_cache"))


@asynccontextmanager
async def browser_context():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--start-maximized",
                "--disable-features=DownloadBubble",
                "--mute-audio",
                "--autoplay-policy=user-gesture-required",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            java_script_enabled=True,
            timezone_id="America/New_York",
            geolocation={"latitude": 40.7128, "longitude": -74.0060},
            permissions=["geolocation"],
            color_scheme="no-preference",
            accept_downloads=False,
        )

        # Add this block to disable autoplay
        await context.add_init_script(
            """
            Object.defineProperty(HTMLMediaElement.prototype, 'autoplay', {
                set: () => {},
                get: () => false
            });
            """
        )

        yield context
        await try_await_or_none(context.close)
        await try_await_or_none(browser.close)


async def open_page(
    context: "BrowserContext",
    url: Union[URL, Text],
    timeout_ms: int = 10000,
    global_wait_seconds: float = 2.0,
) -> Optional["Page"]:
    try:
        page = await context.new_page()
        await page.goto(str(url), wait_until="domcontentloaded", timeout=timeout_ms)
        await simulate_human_behavior(page, timeout_ms=timeout_ms)
        await asyncio.sleep(global_wait_seconds)
        return page
    except PlaywrightTimeoutError:
        console.print_exception()
        return None
    except Exception:
        console.print_exception()
        return None


async def parsing_page(
    url: Union[URL, Text],
    page: Optional["Page"],
    *,
    screenshot_filepath: Optional[Union[Path, Text]] = None,
    cache_result: Cache = cache,
    close_page: bool = True,
    debug: bool = False,
    raise_captcha: bool = False,
    skip_captcha: bool = False,
    manual_solve_captcha: bool = False,  # Default behavior.
) -> Optional[Text]:

    if page is None:
        return None

    content: Optional[Text] = None

    try:
        # Check for CAPTCHA
        if not await handle_captcha_page(
            page,
            raise_captcha=raise_captcha,
            skip_captcha=skip_captcha,
            captcha_manual_solve=manual_solve_captcha,
        ):
            return None

        content = await page.content()
        content = drop_no_used_attrs(drop_all_comments(drop_all_tags(content)))
        if debug:
            debug_print_banner(content, title=f"Content of '{url}'")

        if screenshot_filepath:
            await page.screenshot(
                type="jpeg",
                path=screenshot_filepath,
            )
        cache_result[url] = content

    except PlaywrightTimeoutError:
        console.print_exception()
        if screenshot_filepath:
            await page.screenshot(type="jpeg", path=screenshot_filepath)

    if close_page:
        await try_close_page(page)
    return content


class CrawlerClient:

    def __init__(
        self,
        *,
        cache_dirpath: Optional[Text] = None,
        screenshot_dirpath: Optional[Text] = None,
        debug: bool = False,
    ):
        self.cache_dirpath = cache_dirpath
        self.screenshot_dirpath = screenshot_dirpath
        self.debug = debug

        self.web_cache = Cache(self.get_web_cache_dirpath(cache_dirpath))

    async def search_with_context(
        self,
        query: Text,
        context: "BrowserContext",
        *,
        search_engine: Sequence[Literal["google", "bing", "yahoo", "duckduckgo"]] = (
            "google",
            "bing",
            "yahoo",
            "duckduckgo",
        ),
        num_results: int = 10,
        filter_out_urls: Callable[[Text], bool] = lambda x: False,
        sleep_interval: int = 0,
        raise_search_captcha: bool = False,
        skip_search_captcha: bool = False,
        manual_solve_search_captcha: bool = False,
        skip_url_captcha: bool = True,
        save_failed_url_filepath: Optional[Text] = None,
        filter_empty_content: bool = True,
    ) -> List["HtmlDocument"]:
        """"""

        query = escape_query(query)
        out: List["HtmlDocument"] = []

        # Search google home page with browser
        available_search_engines = get_search_engines(search_engine)
        used_search_engine = random.choice(available_search_engines)
        search_results = await used_search_engine(
            query,
            page=await get_page(context, page_index=0),
            num_results=num_results,
            screenshot_filepath=None,
            cache_result=self.web_cache,
            raise_captcha=raise_search_captcha,
            skip_captcha=skip_search_captcha,
            manual_solve_captcha=manual_solve_search_captcha,
            close_page=False,
        )

        # Filter out URLs
        search_results = [
            _res for _res in search_results if not filter_out_urls(_res.url)
        ]

        # Filter pdf, docx, etc.
        search_results = [
            _res for _res in search_results if not filter_out_extensions(_res.url)
        ]

        # Create HtmlDocument items
        html_docs: List["HtmlDocument"] = [
            HtmlDocument.from_search_result(_res)
            for _res in search_results[:num_results]
        ]
        if not html_docs:
            console.print("No results found.")
            return []

        # Append extra parameters to urls
        for _html_doc in html_docs:
            _html_doc.url = add_extra_params_to_url(_html_doc.url)

        # Request HTML content with multiple pages
        pages = await asyncio.gather(
            *[open_page(context, _html_doc.url) for _html_doc in html_docs]
        )

        for _html_doc, _page in zip(html_docs, pages):
            _html_doc.html_content = await parsing_page(
                _html_doc.url,
                _page,
                screenshot_filepath=None,
                close_page=True,
                skip_captcha=skip_url_captcha,
                cache_result=self.web_cache,
                debug=self.debug,
            )

        # Convert to markdown
        for _html_doc in html_docs:
            if _html_doc.html_content:
                _html_doc.markdown_content = as_markdown(
                    _html_doc.html_content, url=_html_doc.url, debug=self.debug
                )
            else:
                console.print(f"Failed to get content from {_html_doc.url}")
                if save_failed_url_filepath:
                    _failed_url_filepath = Path(save_failed_url_filepath)
                    _failed_url_filepath.touch(exist_ok=True)
                    with open(_failed_url_filepath, "a") as f:
                        f.write(json.dumps({"url": _html_doc.url}) + "\n")

            out.append(_html_doc)

        # Close pages[1:]
        await asyncio.gather(*[try_close_page(_page) for _page in context.pages[1:]])

        if sleep_interval > 0:
            time.sleep(sleep_interval)

        if filter_empty_content:
            out = [_html_doc for _html_doc in out if _html_doc.markdown_content]
        return out

    def get_web_cache_dirpath(self, dirpath: Optional[Text] = None) -> Path:
        if dirpath is None:
            _dirpath = Path.home().joinpath(".languru/data/cache/web_cache")
            _dirpath.mkdir(parents=True, exist_ok=True)
        else:
            _dirpath = Path(dirpath)
        return _dirpath

    def get_img_filepath(
        self, dirpath: Optional[Text] = None, postfix: Optional[Text] = None
    ) -> Path:
        if dirpath is None:
            _dirpath = Path.home().joinpath(".languru/data/screenshot")
            _dirpath.mkdir(parents=True, exist_ok=True)
        else:
            _dirpath = Path(dirpath)
        if postfix is None:
            postfix = str(int(time.time() * 1000))
        return _dirpath.joinpath(f"screenshot-{postfix}.jpg")
