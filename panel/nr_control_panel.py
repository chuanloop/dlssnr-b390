"""Loopback NR control panel: Small/VK102, style, strength and mask.

Writes the shared nr_panel.ini selected by panel-config.json.  Style/intensity/mask are ordinary
per-frame controls; changing them never rebuilds a backend.  Only the backend
selection asks the ASI to reload at the next frame boundary.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import configparser
import json
import os
import secrets
import time
import webbrowser

HERE = Path(__file__).resolve().parent
CONFIG = Path(os.environ.get("NR_PANEL_CONFIG", str(HERE / "panel-config.json")))
settings = json.loads(CONFIG.read_text(encoding="utf-8"))
STATE = Path(settings["state"])
OPTI = Path(settings["optiscaler_ini"])
INI = Path(settings["ini"])
PRIMARY = settings.get("backend", "small")
BACKENDS = dict(settings.get("backends", {}))
BACKENDS[PRIMARY] = settings["manifest"]
if settings.get("vk102_manifest"):
    BACKENDS["102"] = settings["vk102_manifest"]
OPTI_KEYS = ("SpoofedVendorId", "SpoofedDeviceId", "SpoofedGPUName",
             "Dxgi", "StreamlineSpoofing", "SpoofHAGS", "DxgiVRAM")
OPTI_PRESETS = {
    "nvidia4090": {"SpoofedVendorId": "0x10de", "SpoofedDeviceId": "0x2684",
                   "SpoofedGPUName": "NVIDIA GeForce RTX 4090", "Dxgi": "true",
                   "StreamlineSpoofing": "true", "SpoofHAGS": "true", "DxgiVRAM": "auto"},
    "intelb580": {"SpoofedVendorId": "0x8086", "SpoofedDeviceId": "0xe20b",
                  "SpoofedGPUName": "Intel(R) Arc(TM) B580 Graphics", "Dxgi": "false",
                  "StreamlineSpoofing": "auto", "SpoofHAGS": "auto", "DxgiVRAM": "auto"},
    "intelb390": {"SpoofedVendorId": "0x8086", "SpoofedDeviceId": "0xb080",
                  "SpoofedGPUName": "Intel(R) Arc(TM) B390 Graphics", "Dxgi": "false",
                  "StreamlineSpoofing": "auto", "SpoofHAGS": "auto", "DxgiVRAM": "auto"},
}

DEFAULTS = {
    "backend": PRIMARY, "enabled": 1, "style": 0,
    "intensity": 1.0, "local_structure": 1.0, "local_tone": 0.0,
    "skin_structure": 0.0, "auto_mask": 1, "display_strength": 1.0,
}

# --- the Vulkan backend's own knobs -------------------------------------------------------------
# A second, independent control surface.  nr_panel.ini (above) feeds the ASI/handoff path; these
# feed port_network_bridge_backend.dll, which polls nr_tune.txt next to its manifest every 30
# frames.  Two files, two consumers - nothing here can disturb the ASI ABI.
# Ranges mirror the clamps in port_network_bridge_backend.cpp:read_strengths(); a value the panel
# cannot produce is a value the backend clamps anyway, so the slider stops where the clamp starts.
TUNE = Path(settings["manifest"]).parent / "nr_tune.txt"
TUNE_SLIDERS = (
    ("tone",      "光影（低频）",     0.0, 2.0, 0.05),
    ("texture",   "质感（高频）",     0.0, 2.0, 0.05),
    ("detail",    "亮度改动放大",     0.0, 2.0, 0.05),
    ("colour",    "饱和度",           0.0, 4.0, 0.05),
    ("acutance",  "源画面锐度回贴",   0.0, 2.0, 0.05),
    ("guard",     "高光护栏",         1.0, 8.0, 0.5),
    ("intensity", "模型强度 ctl7",    0.0, 1.0, 0.05),
    ("blend",     "端口合成权重",     0.0, 1.0, 0.05),
)
TUNE_SWITCHES = (("delta", "残差合成"), ("easu", "EASU 上采样"),
                 ("fill", "画布填满"), ("dump", "抓一帧"))
TUNE_SCALES = ("auto", "0.33", "0.5", "0.8", "1.0")
TUNE_KEYS = {k for k, *_ in TUNE_SLIDERS} | {k for k, _ in TUNE_SWITCHES} | {"scale"}


def load():
    cfg = configparser.ConfigParser()
    cfg.read(INI, encoding="utf-8")
    out = dict(DEFAULTS)
    if cfg.has_section("NR"):
        for key in out:
            if cfg.has_option("NR", key):
                raw = cfg.get("NR", key)
                if key == "backend":
                    out[key] = raw
                elif key in ("enabled", "style", "auto_mask"):
                    out[key] = int(raw)
                else:
                    out[key] = float(raw)
    return out


def save(values):
    INI.parent.mkdir(parents=True, exist_ok=True)
    cfg = configparser.ConfigParser()
    cfg["NR"] = {
        "backend": values["backend"], "enabled": str(int(values["enabled"])),
        "style": str(int(values["style"])), "intensity": f"{values['intensity']:.4f}",
        "local_structure": f"{values['local_structure']:.4f}",
        "local_tone": f"{values['local_tone']:.4f}",
        "skin_structure": f"{values['skin_structure']:.4f}",
        "auto_mask": str(int(values["auto_mask"])),
        "display_strength": f"{values['display_strength']:.4f}",
    }
    cfg["Backends"] = BACKENDS
    tmp = INI.with_suffix(".tmp")
    try:
        tmp.write_text("", encoding="utf-8")
        with tmp.open("w", encoding="utf-8") as f:
            cfg.write(f)
        tmp.replace(INI)
    except OSError:
        raise


def live_status():
    """Bridge-published state plus the ini the panel owns."""
    out = {"state": None, "ini": load()}
    try:
        out["state"] = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception as exc:  # missing file or a half-written frame
        out["error"] = str(exc)
    return out


def opti_load():
    """Current values of the keys this block owns, comments ignored."""
    out = {key: "auto" for key in OPTI_KEYS}
    try:
        for line in OPTI.read_text(encoding="utf-8-sig").splitlines():
            text = line.strip()
            if not text or text[0] in ";#" or "=" not in text:
                continue
            name, _, value = text.partition("=")
            name = name.strip()
            if name in out:
                out[name] = value.strip()
    except OSError as exc:
        return {"error": str(exc), "values": out}
    return {"values": out, "path": str(OPTI), "presets": OPTI_PRESETS}


def opti_save(values):
    """Rewrite only the owned keys; every other line stays byte-identical."""
    text = OPTI.read_text(encoding="utf-8-sig")
    lines = text.splitlines(True)
    pending = {k: v for k, v in values.items() if k in OPTI_KEYS}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped[0] in ";#" or "=" not in stripped:
            continue
        name = stripped.partition("=")[0].strip()
        if name in pending:
            ending = "\r\n" if line.endswith("\r\n") else "\n"
            lines[index] = "%s = %s%s" % (name, pending.pop(name), ending)
    if pending:
        raise ValueError("unknown keys: " + ", ".join(sorted(pending)))
    if game_running():
        raise ValueError("Close Cyberpunk before changing OptiScaler settings")
    backup = OPTI.with_suffix(".ini.nr-backup")
    if not backup.exists():
        backup.write_bytes(OPTI.read_bytes())
    with OPTI.open("w", encoding="utf-8", newline="") as handle:
        handle.write("".join(lines))
    return opti_load()


def save_async(values):
    import threading
    import time
    def worker():
        for _ in range(80):
            try:
                save(values)
                return
            except OSError:
                time.sleep(0.2)
    threading.Thread(target=worker, daemon=True).start()


def tune_parse(text):
    """key -> raw string, comments and unknown lines left for the caller to keep."""
    out = {}
    for line in text.splitlines():
        head = line.split("#", 1)[0].strip()
        if "=" in head:
            name, _, value = head.partition("=")
            out[name.strip()] = value.strip()
    return out


def tune_backend():
    """What the backend last read, parsed out of nr_tune_read.txt.

    The panel can write any number it likes; only this line proves the DLL saw it. Every re-read
    appends one `dir=... tone=... texture=...` row, so the last row is the live state.
    """
    log = TUNE.parent / "nr_tune_read.txt"
    out = {"log": str(log), "values": {}, "at": None}
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        out["at"] = time.strftime("%H:%M:%S", time.localtime(log.stat().st_mtime))
    except OSError as exc:
        out["error"] = str(exc)
        return out
    last = None
    for line in reversed(lines):
        if line.startswith("dir="):
            last = line
            break
    if last is None:
        out["error"] = "后端还没读过这个文件"
        return out
    pairs = {}
    for token in last.split():
        if "=" in token:
            name, _, value = token.partition("=")
            pairs[name] = value
    out["values"] = {k: pairs.get(k) for k in
                     ("detail", "colour", "guard", "acut", "delta", "scale",
                      "intensity", "blend", "tone", "texture")}
    out["dir"] = pairs.get("dir")
    out["ini"] = pairs.get("ini")
    out["tune"] = pairs.get("tune")
    return out


def tune_load():
    try:
        text = TUNE.read_text(encoding="utf-8")
    except OSError as exc:
        return {"error": str(exc), "path": str(TUNE), "values": {}}
    values = tune_parse(text)
    return {
        "path": str(TUNE),
        "values": values,
        "backend": tune_backend(),
        # The page renders itself from this, so a knob added here shows up without touching the JS.
        "sliders": [{"key": k, "label": label, "min": lo, "max": hi, "step": step}
                    for k, label, lo, hi, step in TUNE_SLIDERS],
        "switches": [{"key": k, "label": label} for k, label in TUNE_SWITCHES],
        "scales": list(TUNE_SCALES),
    }


def tune_save(values):
    try:
        text = TUNE.read_text(encoding="utf-8")
    except OSError:
        text = ""
    bounds = {k: (lo, hi) for k, _label, lo, hi, _step in TUNE_SLIDERS}
    want = {}
    for key, value in values.items():
        if key not in TUNE_KEYS:
            raise ValueError("unknown key: " + str(key))
        if key == "scale":
            text_value = str(value).strip()
            if text_value not in TUNE_SCALES:
                raise ValueError("scale must be one of " + ", ".join(TUNE_SCALES))
            want[key] = text_value
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError("%s: invalid value %r" % (key, value))
        if key in bounds:
            lo, hi = bounds[key]
            number = min(max(number, lo), hi)
        else:                                   # the switches are 0/1
            number = 1.0 if number >= 0.5 else 0.0
        want[key] = ("%g" % number)
    if not want:
        raise ValueError("nothing to write")

    lines, seen = [], set()
    for line in text.splitlines():
        head = line.split("#", 1)[0].strip()
        if "=" in head:
            name = head.partition("=")[0].strip()
            if name in want:
                lines.append("%s=%s" % (name, want[name]))
                seen.add(name)
                continue
        lines.append(line)                      # comments and unrelated keys survive verbatim
    for key, value in want.items():
        if key not in seen:
            lines.append("%s=%s" % (key, value))

    # The backend fopens this file every 30 frames, so a half-written file is a real (if brief)
    # read. Replace in one step: it either sees the old file or the new one.
    tmp = TUNE.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(TUNE)
    return tune_load()


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>NR 后端控制</title><style>
body{font:16px system-ui;background:#101827;color:#eaf2ff;max-width:680px;margin:28px auto;padding:0 16px}
label{display:block;margin:14px 0}select,input{margin-left:10px;background:#26344b;color:#fff;padding:6px}
button{padding:10px 18px;background:#26344b;color:#fff;border:1px solid #537096;border-radius:6px;cursor:pointer}
#live{background:#16233a;border:1px solid #2c3f5c;border-radius:6px;padding:10px 12px;margin:14px 0;font:14px ui-monospace,SFMono-Regular,Consolas,monospace;line-height:1.7}#live span{color:#9fb3cc}#opti{background:#1b2a1f;border:1px solid #35543f;border-radius:6px;padding:10px 12px;margin:14px 0;font:13px ui-monospace,Consolas,monospace;line-height:1.8}#opti .row{margin:6px 0}#opti input{background:#26344b;color:#fff;border:1px solid #3c5270;border-radius:4px;padding:2px 6px;margin-right:8px}#opti small{color:#9fb3cc}.style button{padding:7px 12px;margin-right:6px}.active{background:#2475c7}small{color:#aebed2}#tune{background:#1a2233;border:1px solid #35507a;border-radius:6px;padding:10px 12px;margin:14px 0}#tune .row{display:flex;align-items:center;gap:8px;margin:7px 0}#tune .row>label{flex:0 0 120px;margin:0;font:14px system-ui}#tune .row input[type=range]{flex:1;margin:0}#tune .row output{flex:0 0 48px;text-align:right;font:13px ui-monospace,Consolas,monospace;color:#8ec5ff}#tune .sw{display:inline-flex;align-items:center;gap:5px;margin:4px 16px 4px 0;font:14px system-ui}#tune select{background:#26344b;color:#fff;border:1px solid #3c5270;border-radius:4px;padding:3px 6px}#tune .hint{color:#93a8c4;font-size:12px;line-height:1.5;margin-top:8px}.legacy{background:#2b2318;border:1px solid #6b5424;border-radius:6px;padding:8px 10px;margin:12px 0;font-size:13px;color:#e8c98a;line-height:1.55}.legacy code{background:#3a2f1d;padding:1px 4px;border-radius:3px;font-size:12px}</style><h1>NR 后端控制</h1>
<div id=live>
<div>后端 <b id=lvBackend>—</b> <span id=lvSwitch></span></div>
<div>桥内区间 <b id=lvTotal>—</b> ms <span id=lvStage></span></div>
<div>参数 <span id=lvParams></span></div>
</div><small>桥内区间含跨帧重叠等待，不是游戏帧时。风格/强度/遮罩为逐帧控制；后端切换在帧边界生效。</small>
<div class=legacy><b>合成器区（ASI 路径，VK102 后端不读）</b><br>下面这组写 <code>nr_panel.ini</code>，只作用于 ASI/handoff 那条路。当前游戏走的是 VK102 后端，它读 <code>nr_tune.txt</code>——要改画面请用下方的「网络调参」。</div>
<label>后端 <select id=backend>__BACKEND_OPTIONS__</select></label>
<label>启用 <input id=enabled type=checkbox></label>
<div class=style>风格：<span id=style><button data-v=0>经典</button><button data-v=1>自然</button><button data-v=2>电影</button></span></div>
<label>强度 <input id=intensity type=range min=0 max=2 step=.05><output></output></label>
<label>局部结构 <input id=local_structure type=range min=0 max=2 step=.05><output></output></label>
<label>局部色调 <input id=local_tone type=range min=0 max=2 step=.05><output></output></label>
<label>皮肤/遮罩强度 <input id=skin_structure type=range min=-1 max=2 step=.05><output></output></label>
<label>自动遮罩 <input id=auto_mask type=checkbox></label>
<label>风格强度 <input id=display_strength type=range min=0 max=2 step=.05><output></output></label>
<div id=tune>
<div><b>网络调参</b> <small>（写 nr_tune.txt · 游戏里约 0.5 秒生效 · 不用重启）</small></div>
<div id=tuneSliders></div>
<div id=tuneSwitches></div>
<div class=row>网络档位 <select id=tuneScale></select> <small>auto = 用已装的档；1.0 要静态装，热切会坏</small></div>
<div class=hint>光影/质感是一对拆分：<b>tone=1 texture=0</b> 只剩光影，<b>tone=0 texture=1</b> 只剩质感，1/1 是原始行为。</div>
<div class=row><button id=tuneReset>回 1/1</button><button data-tune="1,0">只看光影</button><button data-tune="0,1">只看质感</button><button data-tune="0,0">全关</button><span id=tuneStatus></span></div>
<div class=hint id=tuneEcho>后端尚未回报</div>
<div class=hint id=tunePath></div>
</div>
<div id=opti>
<div><b>OptiScaler 伪装</b> <small>（写 OptiScaler.ini，重启游戏生效）</small></div>
<div class=row><select id=optiPreset>
<option value=nvidia4090>NVIDIA RTX 4090</option>
<option value=intelb580>Intel B580（原生）</option>
<option value=intelb390>Intel B390（原生）</option>
</select><button id=optiApply>应用预设</button></div>
<div class=row>VendorId <input id=SpoofedVendorId size=8> DeviceId <input id=SpoofedDeviceId size=8></div>
<div class=row>GPU 名 <input id=SpoofedGPUName size=34></div>
<div class=row>DXGI <input id=Dxgi size=6> Streamline <input id=StreamlineSpoofing size=6> HAGS <input id=SpoofHAGS size=6> VRAM <input id=DxgiVRAM size=6></div>
<div class=row><button id=optiSave>保存伪装</button><span id=optiStatus></span></div>
</div>
<button id=save>保存</button><p id=status></p>
<script>
const fields=['enabled','style','intensity','local_structure','local_tone','skin_structure','auto_mask','display_strength'];
let state=null;const token='__TOKEN__';
function set(v){state=v;document.getElementById('backend').value=v.backend;document.getElementById('enabled').checked=!!v.enabled;
 document.getElementById('auto_mask').checked=!!v.auto_mask;
 for(const b of document.querySelectorAll('#style button'))b.classList.toggle('active',+b.dataset.v===v.style);
 for(const k of fields)if(k!=='enabled'&&k!=='style'&&k!=='auto_mask'){const e=document.getElementById(k);e.value=v[k];e.nextElementSibling.value=v[k]}}
async function load(){set(await (await fetch('/api')).json())}
const STYLE_TONE={0:1.0,1:1.0,2:1.0};for(const b of document.querySelectorAll('#style button'))b.onclick=()=>{state.style=+b.dataset.v;state.local_tone=STYLE_TONE[state.style];set(state);document.getElementById('save').click()};
for(const k of ['intensity','local_structure','local_tone','skin_structure','display_strength'])document.getElementById(k).oninput=e=>{state[k]=+e.target.value;e.target.nextElementSibling.value=e.target.value};
document.getElementById('save').onclick=async()=>{state.backend=document.getElementById('backend').value;state.enabled=+document.getElementById('enabled').checked;state.auto_mask=+document.getElementById('auto_mask').checked;
 const r=await fetch('/api',{method:'POST',headers:{'Content-Type':'application/json','X-NR-Token':token},body:JSON.stringify(state)});document.getElementById('status').textContent=r.ok?(poll(),'已保存'):('保存失败: '+((await r.json().catch(()=>({}))).error||'未知'))};
load();
const STYLE_NAME=['经典','自然','电影'];
async function poll(){try{
 const r=await fetch('/status');const d=await r.json();const s=d.state;
 const el=id=>document.getElementById(id);
 if(!s){el('lvBackend').textContent='未连接（bridge 未写状态）';el('lvSwitch').textContent='';
  el('lvTotal').textContent='—';el('lvStage').textContent='';el('lvParams').textContent='';return}
 el('lvBackend').textContent=s.backend+(s.mounted?'（已挂载）':'（未挂载）');
 el('lvSwitch').textContent=s.event?('· '+s.event):'';
 const st=s.stage||{};el('lvTotal').textContent=(st.total||0).toFixed(2);
 el('lvStage').textContent='prep '+(st.prep||0).toFixed(2)+' / host '+(st.host||0).toFixed(2)+
  ' / 合成 '+(st.composite||0).toFixed(2)+' / tail '+(st.tail||0).toFixed(2)+
  ' · 帧 '+(st.frames||0)+' · 计数 '+(s.processed||0);
 const a=s.applied||{};
 el('lvParams').textContent='风格 '+(STYLE_NAME[a.style]||a.style)+' · 强度 '+a.intensity+
  ' · 结构 '+a.structure+' · 色调 '+a.tone+' · 皮肤 '+a.skin+' · 显示 '+a.display+
  ' · 遮罩 '+(a.mask?'开':'关')+' · 启用 '+(a.enabled?'是':'否');
}catch(e){}}
setInterval(poll,250);poll();
async function optiLoad(){const r=await fetch('/opti');const d=await r.json();
 if(d.error){document.getElementById('optiStatus').textContent='读取失败: '+d.error;return}
 const v=d.values||{};for(const k of ['SpoofedVendorId','SpoofedDeviceId','SpoofedGPUName','Dxgi','StreamlineSpoofing','SpoofHAGS','DxgiVRAM']){
  const e=document.getElementById(k);if(e)e.value=v[k]||''}}
async function optiPost(body){const r=await fetch('/opti',{method:'POST',headers:{'Content-Type':'application/json','X-NR-Token':token},body:JSON.stringify(body)});
 const d=await r.json();document.getElementById('optiStatus').textContent=r.ok?'已写入，重启游戏生效':('失败: '+(d.error||''));if(r.ok)optiLoad()}
document.getElementById('optiSave').onclick=()=>{const body={};
 for(const k of ['SpoofedVendorId','SpoofedDeviceId','SpoofedGPUName','Dxgi','StreamlineSpoofing','SpoofHAGS','DxgiVRAM'])
  body[k]=document.getElementById(k).value.trim();optiPost(body)};
document.getElementById('optiApply').onclick=async()=>{const name=document.getElementById('optiPreset').value;
 const r=await fetch('/opti/preset/'+name,{method:'POST',headers:{'X-NR-Token':token}});const d=await r.json();
 document.getElementById('optiStatus').textContent=r.ok?'已套用预设，重启游戏生效':('失败: '+(d.error||''));if(r.ok)optiLoad()};
optiLoad();
let tuneData=null;
function tuneMsg(t){document.getElementById('tuneStatus').textContent=t}
async function tuneLoad(){const r=await fetch('/tune');tuneData=await r.json();
 if(tuneData.error){tuneMsg('读取失败: '+tuneData.error);return}
 document.getElementById('tunePath').textContent='文件：'+(tuneData.path||'');
 const box=document.getElementById('tuneSliders');box.innerHTML='';
 for(const s of tuneData.sliders){
  const row=document.createElement('div');row.className='row';
  const l=document.createElement('label');l.textContent=s.label;
  const i=document.createElement('input');i.type='range';i.min=s.min;i.max=s.max;i.step=s.step;
  const raw=tuneData.values[s.key];i.value=(raw===undefined?1:parseFloat(raw));
  const o=document.createElement('output');o.value=i.value;
  i.oninput=()=>{o.value=i.value};
  i.onchange=()=>tunePost({[s.key]:i.value});
  row.append(l,i,o);box.appendChild(row);
 }
 const sw=document.getElementById('tuneSwitches');sw.innerHTML='';
 for(const s of tuneData.switches){
  const lab=document.createElement('label');lab.className='sw';
  const cb=document.createElement('input');cb.type='checkbox';
  cb.checked=parseFloat(tuneData.values[s.key]||0)>=.5;
  cb.onchange=()=>tunePost({[s.key]:cb.checked?1:0});
  lab.append(cb,document.createTextNode(' '+s.label));sw.appendChild(lab);
 }
 const sel=document.getElementById('tuneScale');sel.innerHTML='';
 for(const v of tuneData.scales){const op=document.createElement('option');op.value=v;op.textContent=v;sel.appendChild(op)}
 sel.value=tuneData.values.scale||'auto';
 sel.onchange=()=>tunePost({scale:sel.value});
 tuneEcho();
}
async function tunePost(body,reflow){
 const r=await fetch('/tune',{method:'POST',headers:{'Content-Type':'application/json','X-NR-Token':token},body:JSON.stringify(body)});
 const d=await r.json();
 if(r.ok){tuneData=d;tuneMsg('已写入，约 0.5 秒生效');tuneEcho();if(reflow)tuneLoad()}
 else{tuneMsg('失败: '+(d.error||''));tuneLoad()}
}
document.getElementById('tuneReset').onclick=()=>tunePost({tone:1,texture:1},1);
for(const b of document.querySelectorAll('[data-tune]'))b.onclick=()=>{
 const p=b.dataset.tune.split(',');tunePost({tone:+p[0],texture:+p[1]},1);
};
function tuneEcho(){
 const b=tuneData.backend||{};const el=document.getElementById('tuneEcho');
 if(b.error){el.textContent='后端：'+b.error;el.style.color='#e0a04a';return}
 const v=b.values||{},mine=tuneData.values||{};
 const pick=k=>(v[k]===undefined||v[k]===null)?'—':v[k];
 const stale=[];
 for(const k of ['tone','texture']){
  if(v[k]===undefined||v[k]===null)continue;
  if(Math.abs(parseFloat(v[k])-parseFloat(mine[k]||1))>0.011)stale.push(k);
 }
 el.textContent='后端 '+b.at+' 已读 · tone '+pick('tone')+' · texture '+pick('texture')+
  ' · delta '+pick('delta')+' · scale '+pick('scale')+
  (stale.length?'   ← '+stale.join('/')+' 还没生效':'');
 el.style.color=stale.length?'#e0a04a':'#7fc08a';
}
setInterval(async()=>{if(!tuneData)return;
 try{const r=await fetch('/tune');const d=await r.json();if(d.backend)tuneData.backend=d.backend;tuneEcho()}catch(e){}},2000);
tuneLoad();
</script>"""


import sys
import threading
import time

WATCH = "--watch" in sys.argv
SERVER = None


def game_running():
    import subprocess
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Cyberpunk2077.exe"],
                             capture_output=True, text=True, timeout=5).stdout
        return "Cyberpunk2077.exe" in out
    except Exception:
        return False


def watch_game():
    # Started by the launcher before the game: wait for it, then exit with it.
    deadline = time.time() + 180
    while time.time() < deadline and not game_running():
        time.sleep(2)
    while game_running():
        time.sleep(5)
    if SERVER is not None:
        SERVER.shutdown()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, data, ctype="application/json"):
        body = data.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            options = ''.join(f'<option value={key}>{label}</option>' for key, label in
                              (("small", "Small 310.9.16"), ("102", "VK102 310.8")) if key in BACKENDS)
            page = PAGE.replace("__BACKEND_OPTIONS__", options)
            self._send(200, page.replace("__TOKEN__", TOKEN), "text/html; charset=utf-8")
        elif self.path == "/api":
            self._send(200, json.dumps(load()))
        elif self.path == "/status":
            self._send(200, json.dumps(live_status()))
        elif self.path == "/opti":
            self._send(200, json.dumps(opti_load()))
        elif self.path == "/tune":
            self._send(200, json.dumps(tune_load()))
        elif self.path.startswith("/opti/preset/"):
            self._send(404, "{}")
        else:
            self._send(404, "{}")

    def do_POST(self):
        if self.headers.get("X-NR-Token") != TOKEN:
            return self._send(403, "{}")
        if self.path.startswith("/opti/preset/"):
            name = self.path.rsplit("/", 1)[-1]
            try:
                preset = OPTI_PRESETS.get(name)
                if not preset:
                    raise ValueError("unknown preset " + name)
                self._send(200, json.dumps(opti_save(preset)))
            except Exception as exc:
                self._send(400, json.dumps({"error": str(exc)}))
            return
        if self.path == "/opti":
            try:
                values = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                self._send(200, json.dumps(opti_save(values)))
            except Exception as exc:
                self._send(400, json.dumps({"error": str(exc)}))
            return
        if self.path == "/tune":
            try:
                values = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                self._send(200, json.dumps(tune_save(values)))
            except Exception as exc:
                self._send(400, json.dumps({"error": str(exc)}))
            return
        if self.path != "/api":
            return self._send(403, "{}")
        try:
            values = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            merged = load()
            for key in DEFAULTS:
                if key not in values:
                    continue
                value = values[key]
                if value is None:      # a stale page may post null
                    continue
                try:
                    if key == "backend":
                        merged[key] = str(value)
                    elif key in ("enabled", "style", "auto_mask"):
                        merged[key] = int(value)
                    else:
                        merged[key] = float(value)
                except (TypeError, ValueError):
                    raise ValueError("%s: invalid value %r" % (key, value))
            save(merged)   # synchronous: a failed write must be visible
            self._send(200, json.dumps(merged))
        except Exception as exc:
            self._send(400, json.dumps({"error": str(exc)}))


TOKEN = secrets.token_hex(24)
if __name__ == "__main__":
    if not INI.exists():
        save(dict(DEFAULTS))
    # A busy port used to end the panel with a bare traceback, which reads as
    # "the panel does not open". Walk forward instead and print the address that
    # actually bound, so a stale instance on the default port cannot block it.
    first = int(os.environ.get("NR_PANEL_PORT", "8788"))
    server = None
    for offset in range(0, 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", first + offset), Handler)
            port = first + offset
            if offset:
                print(f"port {first} busy, using {port}")
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"no free port in {first}..{first + 9}")
    SERVER = server
    url = f"http://127.0.0.1:{port}/"
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    try:
        print(f"{url}  (config: {INI})")
    except Exception:
        pass
    webbrowser.open(url)
    if WATCH:
        threading.Thread(target=watch_game, daemon=True).start()
    server.serve_forever()
