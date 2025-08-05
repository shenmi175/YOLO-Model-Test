import os
import json
import argparse
from flask import Flask, send_from_directory, request, jsonify, render_template_string

app = Flask(__name__)
IMAGE_DIR = ""

# Connections between keypoints including virtual point 16
CONNECTIONS = [
    (4, 2), (2, 0), (1, 0), (1, 3),
    (6, 8), (8, 10), (5, 7), (7, 9),
    (12, 14), (14, 16), (11, 13), (13, 15),
    (5, 16), (6, 16), (12, 16), (11, 16)
]


@app.route("/")
def index():
    images = [f for f in os.listdir(IMAGE_DIR)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    items = "".join(
        f'<li><a href="/annotate/{img}">{img}</a></li>' for img in images
    )
    return f"<h1>Select Image</h1><ul>{items}</ul>"


@app.route("/images/<path:filename>")
def image_file(filename):
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/load/<image>")
def load(image):
    path = os.path.join(IMAGE_DIR, os.path.splitext(image)[0] + ".json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    return jsonify(data)


@app.route("/save/<image>", methods=["POST"])
def save(image):
    path = os.path.join(IMAGE_DIR, os.path.splitext(image)[0] + ".json")
    data = request.get_json(force=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return jsonify({"success": True})


@app.route("/annotate/<image>")
def annotate(image):
    template = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Keypoint Annotation</title>
<style>
body {margin:0; display:flex;}
#panel {width:200px; background:#f0f0f0; padding:10px; overflow:auto;}
#canvas {cursor:crosshair;}
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div id="panel">
<h3>Labels</h3>
<ul id="list"></ul>
</div>
<script>
const imagePath = "{{ url_for('image_file', filename=image) }}";
const loadUrl = "{{ url_for('load', image=image) }}";
const saveUrl = "{{ url_for('save', image=image) }}";
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let img = new Image();
let points = [];
let selected = null;
const connections = {{ connections }};

img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    loadData();
};
img.src = imagePath;

function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(img,0,0);
    // compute virtual point 16
    const p5 = points.find(p=>p.label==='5');
    const p6 = points.find(p=>p.label==='6');
    let p16 = points.find(p=>p.label==='16');
    if(p5 && p6){
        const cx = (p5.x+p6.x)/2;
        const cy = (p5.y+p6.y)/2;
        if(p16){ p16.x = cx; p16.y = cy; }
        else { points.push({label:'16', x:cx, y:cy}); }
    }
    ctx.strokeStyle='lime';
    connections.forEach(([a,b])=>{
        const pa = points.find(p=>p.label==String(a));
        const pb = points.find(p=>p.label==String(b));
        if(pa && pb){
            ctx.beginPath();
            ctx.moveTo(pa.x,pa.y);
            ctx.lineTo(pb.x,pb.y);
            ctx.stroke();
        }
    });
    points.forEach(p=>{
        ctx.fillStyle = selected && selected.label===p.label ? 'yellow':'red';
        ctx.beginPath();
        ctx.arc(p.x,p.y,5,0,Math.PI*2);
        ctx.fill();
    });
    updateList();
    saveData();
}

function updateList(){
    const list = document.getElementById('list');
    list.innerHTML='';
    points.forEach(p=>{
        const li=document.createElement('li');
        li.textContent=p.label;
        li.onclick=()=>{selected=p; draw();};
        list.appendChild(li);
    });
}

canvas.addEventListener('dblclick', e=>{
    const rect=canvas.getBoundingClientRect();
    const x=e.clientX-rect.left;
    const y=e.clientY-rect.top;
    const label=prompt('label?');
    if(!label) return;
    points=points.filter(p=>p.label!==label);
    points.push({label:label,x:x,y:y});
    draw();
});

canvas.addEventListener('click', e=>{
    const rect=canvas.getBoundingClientRect();
    const x=e.clientX-rect.left;
    const y=e.clientY-rect.top;
    selected=null;
    points.forEach(p=>{
        if(Math.hypot(p.x-x,p.y-y)<6){selected=p;}
    });
    draw();
});

document.addEventListener('keydown', e=>{
    if(e.key==='Delete' && selected){
        points = points.filter(p=>p!==selected);
        selected=null;
        draw();
    }
});

function loadData(){
    fetch(loadUrl).then(r=>r.json()).then(data=>{
        if(data.shapes){
            data.shapes.forEach(s=>{
                if(s.shape_type==='point'){
                    points.push({label:s.label,x:s.points[0][0],y:s.points[0][1]});
                }
            });
        }
        draw();
    });
}

function saveData(){
    const shapes = points.map(p=>({label:p.label, points:[[p.x,p.y]], group_id:1, shape_type:'point', flags:{}}));
    const payload = {version:'5.0.1', flags:{}, shapes:shapes, imagePath:'{{ image }}', imageHeight:img.height, imageWidth:img.width};
    fetch(saveUrl,{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
}
</script>
</body>
</html>
"""
    return render_template_string(template, image=image, connections=CONNECTIONS)


def main():
    parser = argparse.ArgumentParser(description="Keypoint annotation web tool")
    parser.add_argument("dir", help="directory with images and json files")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    global IMAGE_DIR
    IMAGE_DIR = args.dir
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()