# -*- coding: utf-8 -*-
"""
DeepSeek AI 缺陷分析网站 —— 本地代理服务（零依赖，仅用 Python 标准库）

作用：
  1. 在 http://127.0.0.1:8765 上托管网页（index.html）
  2. 提供 /api/chat 接口，把浏览器的请求转发给 DeepSeek 云端 API，
     解决浏览器直接调用 DeepSeek 的跨域(CORS)问题，并避免把 API Key 暴露给第三方页面。

启动：双击 start.bat，或在此目录执行  python server.py
停止：在黑色窗口里按 Ctrl+C
"""
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# 仅本机访问（如需让树莓派/局域网其他设备访问，改为 HOST = "0.0.0.0" 并放行防火墙端口）
HOST = "127.0.0.1"
PORT = 8766
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_API_BASE = "https://api.deepseek.com"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_api_key():
    """API Key 的后备来源：环境变量 DEEPSEEK_API_KEY > 同目录 config.json。
    正常使用在网页"设置"里填即可（存在浏览器本地），这里只是多一层后备。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    cfg = os.path.join(BASE_DIR, "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            k = str(data.get("api_key", "")).strip()
            if k:
                return k
        except Exception:
            pass
    return ""


class Handler(BaseHTTPRequestHandler):
    server_version = "DeepSeekAIWeb/1.0"

    def log_message(self, fmt, *args):
        sys.stdout.write("[server] " + (fmt % args) + "\n")
        sys.stdout.flush()

    # ---------------- 静态页面 ----------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._serve_file(os.path.join(BASE_DIR, "index.html"),
                             "text/html; charset=utf-8")
            return
        if path == "/health":
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # 拉取树莓派页面（绕开浏览器跨域）
        if path == "/api/pi-data":
            self._handle_pi_data()
            return
        # 其它同目录静态文件（防目录穿越）
        rel = path.lstrip("/")
        full = os.path.normpath(os.path.join(BASE_DIR, rel))
        if not full.startswith(BASE_DIR) or not os.path.isfile(full):
            self.send_error(404, "Not Found")
            return
        if full.endswith(".html"):
            ctype = "text/html; charset=utf-8"
        elif full.endswith(".js"):
            ctype = "text/javascript; charset=utf-8"
        elif full.endswith(".css"):
            ctype = "text/css; charset=utf-8"
        elif full.endswith(".json"):
            ctype = "application/json; charset=utf-8"
        elif full.endswith(".png"):
            ctype = "image/png"
        elif full.endswith(".svg"):
            ctype = "image/svg+xml"
        else:
            ctype = "application/octet-stream"
        self._serve_file(full, ctype)

    def _serve_file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception:
            self.send_error(404, "Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # ---------------- AI 聊天代理 ----------------
    def do_POST(self):
        if self.path.split("?")[0] != "/api/chat":
            self.send_error(404, "Not Found")
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._json_error(400, "请求体不是合法的 JSON")
            return

        api_key = str(payload.get("api_key") or get_api_key() or "").strip()
        if not api_key:
            self._json_error(
                400,
                "未设置 API Key：请点击网页右上角“设置”，填入 DeepSeek API Key。"
                "（也可在 server.py 同目录放 config.json：{\"api_key\":\"sk-...\"}）",
            )
            return

        base = str(payload.get("base_url") or DEFAULT_API_BASE).strip().rstrip("/")
        model = str(payload.get("model") or "deepseek-chat").strip()
        try:
            temperature = float(payload.get("temperature", 0.3))
        except (TypeError, ValueError):
            temperature = 0.3
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            self._json_error(400, "messages 不能为空")
            return
        stream = bool(payload.get("stream", True))

        upstream = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if payload.get("max_tokens"):
            try:
                upstream["max_tokens"] = int(payload["max_tokens"])
            except (TypeError, ValueError):
                pass

        req = Request(
            base + "/chat/completions",
            data=json.dumps(upstream).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key,
            },
            method="POST",
        )
        try:
            resp = urlopen(req, timeout=300)
        except HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            hint = ""
            if e.code == 401:
                hint = ("【401 = API Key 无效】Key 输错、已被删除/吊销或已过期，"
                        "请到 https://platform.deepseek.com → API Keys 重新创建并复制。")
            elif e.code == 402:
                hint = "【402 = 余额不足】请到 https://platform.deepseek.com → 充值。"
            elif e.code == 404:
                hint = "【404 = 接口地址不对】请检查设置里的“接口地址”（应为 https://api.deepseek.com）。"
            self._json_error(e.code, "DeepSeek 接口返回错误 " + str(e.code) + "：" + detail + hint)
            return
        except URLError as e:
            self._json_error(502, "无法连接 DeepSeek（" + str(e.reason) + "），请检查网络或接口地址")
            return
        except Exception as e:
            self._json_error(500, "请求异常：" + str(e))
            return

        if not stream:
            data = resp.read()
            ctype = resp.headers.get("Content-Type", "application/json")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        # 流式：把 DeepSeek 返回的 SSE 逐行转发给浏览器
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                self.wfile.write(line)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                resp.close()
            except Exception:
                pass

    def _json(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, code, msg):
        self._json(code, json.dumps({"error": msg}, ensure_ascii=False).encode("utf-8"))

    PINYIN_MAP = {
        "ca_shang": "擦伤", "hua_shang": "划伤", "liebian": "裂边", "zang_wu": "脏污",
        "zhe_zhou": "褶皱", "zhen_kong": "针孔", "zheng_chang": "正常", "other": "其他",
    }

    def _get_pi_url(self, query_url):
        """取树莓派地址：优先接口地址 ?url=，其次 config.json 的 pi_url"""
        url = str(query_url or "").strip()
        if url:
            return url
        cfg = os.path.join(BASE_DIR, "config.json")
        if os.path.isfile(cfg):
            try:
                with open(cfg, "r", encoding="utf-8") as f:
                    url = str(json.load(f).get("pi_url", "")).strip()
            except Exception:
                pass
        return url

    def _http_get_text(self, url, timeout=5):
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="GET")
        resp = urlopen(req, timeout=timeout)
        raw = resp.read()
        for enc in ("utf-8", "gbk", "gb18030", "latin-1"):
            try:
                return raw.decode(enc)
            except Exception:
                continue
        return raw.decode("utf-8", "replace")

    def _handle_pi_data(self):
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        url = self._get_pi_url((qs.get("url") or [""])[0])
        if not url:
            self._json_error(
                400,
                "未提供树莓派地址：请在页面设置里填写“树莓派数据地址”，"
                "或在 config.json 里写 {\"pi_url\":\"http://树莓派IP:端口\"}",
            )
            return
        base = url.rstrip("/")

        # 1) 候选后端：用户给的地址本身 + 其 config.json 里声明的后端端口
        candidates = [base]
        try:
            cfg_text = self._http_get_text(base + "/config.json", timeout=5)
            cfg = json.loads(cfg_text)
            for field in ("url", "ip"):
                v = str(cfg.get(field) or "").strip()
                if v:
                    candidates.append(v.rstrip("/"))
        except Exception:
            pass
        # 去重保序
        seen, uniq = set(), []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        candidates = uniq

        # 2) 找能返回 /get_num 的后端
        backend = None
        num_text = None
        for cand in candidates:
            try:
                text = self._http_get_text(cand + "/get_num", timeout=5)
                if text.strip() and "[" in text:
                    backend, num_text = cand, text
                    break
            except Exception:
                continue
        if not backend:
            self._json_error(
                502,
                "在以下地址都没找到数据接口 /get_num：" + "、".join(candidates)
                + "。请确认树莓派检测程序已运行（7777 端口服务）",
            )
            return

        # 3) 标签映射：/get_conf 的 defect_name 优先，拼音兜底
        label_map = dict(self.PINYIN_MAP)
        try:
            conf = json.loads(self._http_get_text(backend + "/get_conf", timeout=5))
            if isinstance(conf, list) and conf and isinstance(conf[0], dict):
                dn = conf[0].get("defect_name")
                if isinstance(dn, dict):
                    for k, v in dn.items():
                        label_map[str(k)] = str(v)
        except Exception:
            pass

        # 4) 解析 /get_num 里的 {键: 数量}（可能是任意嵌套结构）
        try:
            num = json.loads(num_text)
        except Exception:
            self._json_error(502, "后端返回的 /get_num 不是合法 JSON：" + (num_text or "")[:200])
            return
        entries = {}
        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        entries[str(k)] = int(v)
                    else:
                        walk(v)
            elif isinstance(x, list):
                for i in x:
                    walk(i)
        walk(num)
        if not entries:
            self._json_error(502, "/get_num 中没有识别出计数数据：" + (num_text or "")[:200])
            return

        items = [{"key": k, "label": label_map.get(k, k), "value": v} for k, v in entries.items()]
        self._json(200, json.dumps(
            {"ok": True, "backend": backend, "items": items},
            ensure_ascii=False,
        ).encode("utf-8"))


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = "http://127.0.0.1:%d" % PORT
    print("=" * 56)
    print("  DeepSeek AI 缺陷分析网站 已启动")
    print("  请在浏览器打开：" + url)
    print("  按 Ctrl+C 停止服务")
    print("=" * 56)
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
