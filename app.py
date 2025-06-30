# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from pathlib import Path
import subprocess, shutil, os
import cv2
import numpy as np
import tempfile
from typing import List

app = FastAPI()

# Adicionar CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path("C:/Imagens/3d")  # Caminho para Windows
COLMAP_BIN = "C:/Users/willi/Downloads/colmap/COLMAP.bat"  # Arquivo batch do COLMAP

@app.post("/reconstruir")
async def reconstruir_modelo(files: list[UploadFile] = File(None), video: UploadFile = File(None)):
    """Endpoint unificado para reconstrução 3D a partir de imagens ou vídeo"""
    
    if video is not None:
        # Processamento de vídeo
        return await processar_video(video)
    elif files and len(files) > 0:
        # Processamento de imagens
        return await processar_imagens(files)
    else:
        return JSONResponse(status_code=400, content={
            "error": "Nenhum arquivo foi enviado. Envie imagens ou um vídeo."
        })


async def processar_video(video: UploadFile):
    """Processa vídeo para reconstrução 3D"""
    print(f"\n=== INÍCIO DO PROCESSAMENTO DE VÍDEO ===")
    print(f"Arquivo recebido: {video.filename}")
    print(f"Content type: {video.content_type}")
    
    job_id = str(uuid4())
    job_path = BASE_DIR / job_id
    
    # Detectar extensão do vídeo
    file_extension = ".mp4"  # padrão
    if video.filename:
        file_extension = os.path.splitext(video.filename)[1] or ".mp4"
    
    video_path = job_path / f"input_video{file_extension}"
    image_path = job_path / "images"
    sparse_path = job_path / "sparse"
    output_model = job_path / "model.ply"
    db_path = job_path / "database.db"

    print(f"Job ID: {job_id}")
    print(f"Caminho do vídeo: {video_path}")

    job_path.mkdir(parents=True, exist_ok=True)
    image_path.mkdir(parents=True, exist_ok=True)
    sparse_path.mkdir(parents=True, exist_ok=True)

    try:
        # Salva o vídeo
        print(f"Salvando vídeo: {video.filename}")
        content = await video.read()
        print(f"Conteúdo lido: {len(content)} bytes")
        
        if len(content) == 0:
            print("ERRO: Arquivo de vídeo vazio!")
            return JSONResponse(status_code=400, content={
                "error": "Arquivo de vídeo está vazio"
            })
        
        with open(video_path, "wb") as f:
            f.write(content)
        
        # Verificar se o arquivo foi salvo corretamente
        saved_size = os.path.getsize(video_path)
        print(f"Arquivo salvo com {saved_size} bytes")
        
        # Testar se OpenCV consegue abrir o arquivo
        print("Testando abertura com OpenCV...")
        test_cap = cv2.VideoCapture(str(video_path))
        if not test_cap.isOpened():
            print("ERRO: OpenCV não conseguiu abrir o vídeo")
            return JSONResponse(status_code=400, content={
                "error": "Formato de vídeo não suportado. Tente converter para MP4."
            })
        
        test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        test_fps = test_cap.get(cv2.CAP_PROP_FPS)
        print(f"OpenCV conseguiu abrir: {test_frames} frames, {test_fps} FPS")
        test_cap.release()
        
        # Extrai os melhores frames
        print("Extraindo frames inteligentemente...")
        frame_count = extract_best_frames(str(video_path), image_path, target_frames=50)  # Mais frames para melhor textura
        print(f"Frames extraídos: {frame_count}")
        
        # Verificar se frames foram realmente salvos
        saved_frames = list(image_path.glob("*.jpg"))
        print(f"Arquivos de frames encontrados: {len(saved_frames)}")
        
        if len(saved_frames) < 15:  # Threshold mais baixo para permitir reconstrução densa
            return JSONResponse(status_code=400, content={
                "error": f"Poucos frames válidos extraídos ({len(saved_frames)}). Tente um vídeo com melhor qualidade."
            })

        print(f"Processando {len(saved_frames)} frames com COLMAP...")
        
        # Executa pipeline COLMAP
        await executar_pipeline_colmap(db_path, image_path, sparse_path, output_model)

        # Remove vídeo temporário para economizar espaço
        if video_path.exists():
            video_path.unlink()
            print("Vídeo temporário removido")

        print(f"=== PROCESSAMENTO CONCLUÍDO COM SUCESSO ===\n")
        return {"status": "ok", "url": f"/visualizar/{job_id}", "frames_extracted": len(saved_frames)}

    except Exception as e:
        print(f"Erro no processamento de vídeo: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


async def processar_imagens(files: list[UploadFile]):
    """Processa imagens para reconstrução 3D"""
    job_id = str(uuid4())
    job_path = BASE_DIR / job_id
    image_path = job_path / "images"
    sparse_path = job_path / "sparse"
    output_model = job_path / "model.ply"
    db_path = job_path / "database.db"

    image_path.mkdir(parents=True, exist_ok=True)
    sparse_path.mkdir(parents=True, exist_ok=True)

    # Salva as imagens
    for i, file in enumerate(files):
        with open(image_path / f"img_{i}.jpg", "wb") as f:
            f.write(await file.read())

    try:
        # Executa pipeline COLMAP
        await executar_pipeline_colmap(db_path, image_path, sparse_path, output_model)
        return {"status": "ok", "url": f"/visualizar/{job_id}"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


async def executar_pipeline_colmap(db_path, image_path, sparse_path, output_model):
    """Executa o pipeline completo do COLMAP com geração de malha texturizada"""
    try:
        # Caminhos para diferentes formatos de saída
        job_path = output_model.parent
        dense_path = job_path / "dense"
        fused_path = dense_path / "fused.ply"
        meshed_path = dense_path / "meshed-poisson.ply"
        textured_path = job_path / "textured"
        
        dense_path.mkdir(exist_ok=True)
        textured_path.mkdir(exist_ok=True)
        
        # COLMAP pipeline básico
        print(f"Iniciando feature extraction para imagens em {image_path}...")
        subprocess.run([COLMAP_BIN, "feature_extractor",
                        "--database_path", str(db_path),
                        "--image_path", str(image_path)], 
                        check=True, capture_output=True, text=True)
        
        print("Feature extraction concluída. Iniciando matching...")
        subprocess.run([COLMAP_BIN, "exhaustive_matcher",
                        "--database_path", str(db_path)], 
                        check=True, capture_output=True, text=True)

        print("Matching concluído. Iniciando mapping...")
        subprocess.run([COLMAP_BIN, "mapper",
                        "--database_path", str(db_path),
                        "--image_path", str(image_path),
                        "--output_path", str(sparse_path)], 
                        check=True, capture_output=True, text=True)

        # Verifica se a reconstrução esparsa foi bem-sucedida
        model_bin_path = sparse_path / "0"
        if not model_bin_path.exists() or not (model_bin_path / "points3D.bin").exists():
            raise Exception("Reconstrução esparsa falhou - modelo não gerado")

        print("Mapping concluído. Iniciando reconstrução densa...")
        
        # Undistort images para dense reconstruction
        subprocess.run([COLMAP_BIN, "image_undistorter",
                        "--image_path", str(image_path),
                        "--input_path", str(model_bin_path),
                        "--output_path", str(dense_path),
                        "--output_type", "COLMAP"], 
                        check=True, capture_output=True, text=True)

        # Patch match stereo
        print("Executando stereo matching...")
        subprocess.run([COLMAP_BIN, "patch_match_stereo",
                        "--workspace_path", str(dense_path),
                        "--workspace_format", "COLMAP",
                        "--PatchMatchStereo.geom_consistency", "true"], 
                        check=True, capture_output=True, text=True)

        # Stereo fusion
        print("Executando stereo fusion...")
        subprocess.run([COLMAP_BIN, "stereo_fusion",
                        "--workspace_path", str(dense_path),
                        "--workspace_format", "COLMAP",
                        "--input_type", "geometric",
                        "--output_path", str(fused_path)], 
                        check=True, capture_output=True, text=True)

        # Poisson meshing
        print("Gerando malha com Poisson...")
        try:
            subprocess.run([COLMAP_BIN, "poisson_mesher",
                            "--input_path", str(fused_path),
                            "--output_path", str(meshed_path)], 
                            check=True, capture_output=True, text=True)
            print(f"Malha Poisson gerada: {meshed_path}")
        except subprocess.CalledProcessError as e:
            print(f"Poisson meshing falhou, usando nuvem de pontos: {e}")
            meshed_path = fused_path

        # Converte modelo para formatos web-friendly
        print("Convertendo modelos para visualização...")
        
        # Converte para PLY (fallback)
        ply_output = job_path / "model.ply"
        if meshed_path.exists():
            shutil.copy(meshed_path, ply_output)
            print(f"Modelo PLY copiado: {ply_output}")
        elif fused_path.exists():
            shutil.copy(fused_path, ply_output)
            print(f"Nuvem de pontos copiada: {ply_output}")
        
        # Tenta converter para OBJ com texturas
        obj_output = job_path / "model.obj"
        try:
            if model_bin_path.exists():
                subprocess.run([COLMAP_BIN, "model_converter",
                               "--input_path", str(model_bin_path),
                               "--output_path", str(obj_output),
                               "--output_type", "OBJ"], 
                               check=True, capture_output=True, text=True)
                print(f"Modelo OBJ gerado: {obj_output}")
        except subprocess.CalledProcessError as e:
            print(f"Conversão OBJ falhou: {e}")

        # Verifica se pelo menos um modelo foi gerado
        if not ply_output.exists():
            raise Exception("Nenhum modelo foi gerado com sucesso")

    except subprocess.CalledProcessError as e:
        error_info = {
            "error": f"COLMAP falhou: {str(e)}",
            "command": " ".join(e.cmd),
            "return_code": e.returncode,
            "stdout": e.stdout if hasattr(e, 'stdout') and e.stdout else "Sem stdout",
            "stderr": e.stderr if hasattr(e, 'stderr') and e.stderr else "Sem stderr"
        }
        print(f"Erro COLMAP: {error_info}")
        raise Exception(f"COLMAP falhou: {error_info['error']}")


@app.get("/visualizar/{job_id}")
def visualizar_modelo(job_id: str):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Visualização 3D - {job_id}</title>
        <style>
            body {{ margin: 0; background: #000; color: white; font-family: Arial; }}
            #info {{ position: absolute; top: 10px; left: 10px; z-index: 100; padding: 10px; background: rgba(0,0,0,0.8); border-radius: 5px; max-width: 300px; }}
            #controls {{ position: absolute; top: 10px; right: 10px; z-index: 100; padding: 10px; background: rgba(0,0,0,0.8); border-radius: 5px; }}
            .error {{ color: #ff4444; }}
            .success {{ color: #44ff44; }}
            .warning {{ color: #ffff44; }}
            canvas {{ display: block; }}
            button {{ margin: 5px; padding: 8px 12px; background: #333; color: white; border: 1px solid #555; border-radius: 4px; cursor: pointer; }}
            button:hover {{ background: #555; }}
            button.active {{ background: #0066cc; }}
            select {{ margin: 5px; padding: 5px; background: #333; color: white; border: 1px solid #555; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div id="info">Carregando informações do modelo...</div>
        <div id="controls">
            <div>
                <button id="btnPoints" class="active" onclick="toggleRenderMode('points')">Pontos</button>
                <button id="btnMesh" onclick="toggleRenderMode('mesh')">Malha</button>
                <button id="btnWireframe" onclick="toggleRenderMode('wireframe')">Wireframe</button>
            </div>
            <div>
                <label>Tamanho dos Pontos:</label>
                <input type="range" id="pointSize" min="0.1" max="5" step="0.1" value="1" onchange="updatePointSize(this.value)">
            </div>
            <div>
                <button onclick="resetCamera()">Resetar Câmera</button>
                <button onclick="toggleBackground()">Alternar Fundo</button>
            </div>
        </div>
        <canvas id="c"></canvas>
    
    <script type="importmap">
        {{
            "imports": {{
                "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
                "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
            }}
        }}
    </script>
    
    <script type="module">
        import * as THREE from 'three';
        import {{ PLYLoader }} from 'three/addons/loaders/PLYLoader.js';
        import {{ OBJLoader }} from 'three/addons/loaders/OBJLoader.js';
        import {{ MTLLoader }} from 'three/addons/loaders/MTLLoader.js';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
        
        console.log('Iniciando visualização 3D avançada...');
        
        const info = document.getElementById('info');
        let currentModel = null;
        let currentRenderMode = 'points';
        let modelInfo = null;
        let originalMaterial = null;
        let darkBackground = true;
        
        // Função para mostrar status
        function showStatus(message, type = 'success') {{
            info.innerHTML = message;
            info.className = type;
            console.log(message);
        }}
        
        // Configurar cena
        const canvas = document.getElementById('c');
        const renderer = new THREE.WebGLRenderer({{canvas, antialias: true}});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor(0x222222);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
        
        // Adicionar controles de órbita
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.1;
        
        // Adicionar luzes
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        scene.add(directionalLight);
        
        const pointLight = new THREE.PointLight(0xffffff, 0.4);
        pointLight.position.set(-10, -10, -5);
        scene.add(pointLight);
        
        // Funções globais para controles
        window.toggleRenderMode = function(mode) {{
            currentRenderMode = mode;
            document.querySelectorAll('#controls button').forEach(btn => btn.classList.remove('active'));
            document.getElementById('btn' + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add('active');
            
            if (currentModel) {{
                updateModelDisplay();
            }}
        }};
        
        window.updatePointSize = function(size) {{
            if (currentModel && currentModel.material && currentModel.material.size !== undefined) {{
                currentModel.material.size = parseFloat(size);
            }}
        }};
        
        window.resetCamera = function() {{
            if (currentModel) {{
                const box = new THREE.Box3().setFromObject(currentModel);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                
                camera.position.set(maxDim, maxDim, maxDim);
                camera.lookAt(center);
                controls.target.copy(center);
                controls.update();
            }}
        }};
        
        window.toggleBackground = function() {{
            darkBackground = !darkBackground;
            renderer.setClearColor(darkBackground ? 0x222222 : 0xf0f0f0);
        }};
        
        function updateModelDisplay() {{
            if (!currentModel) return;
            
            if (currentModel.type === 'Points') {{
                // Modelo de pontos
                currentModel.visible = currentRenderMode === 'points';
            }} else if (currentModel.type === 'Group' || currentModel.type === 'Mesh') {{
                // Modelo OBJ (malha)
                currentModel.traverse(function(child) {{
                    if (child.isMesh) {{
                        child.visible = currentRenderMode !== 'points';
                        
                        if (currentRenderMode === 'wireframe') {{
                            child.material.wireframe = true;
                        }} else {{
                            child.material.wireframe = false;
                        }}
                    }}
                }});
            }}
        }}
        
        // Carrega informações do modelo
        async function loadModelInfo() {{
            try {{
                const response = await fetch('/modelo/{job_id}/info');
                modelInfo = await response.json();
                console.log('Model info:', modelInfo);
                
                if (modelInfo.has_obj) {{
                    showStatus('Carregando modelo 3D com texturas...', 'success');
                    await loadOBJModel();
                }} else if (modelInfo.has_ply) {{
                    showStatus('Carregando nuvem de pontos...', 'warning');
                    await loadPLYModel();
                }} else {{
                    showStatus('Nenhum modelo encontrado', 'error');
                }}
            }} catch (error) {{
                console.error('Erro ao carregar info:', error);
                showStatus('Tentando carregar modelo PLY...', 'warning');
                await loadPLYModel();
            }}
        }}
        
        // Carrega modelo OBJ com texturas
        async function loadOBJModel() {{
            try {{
                let materials = null;
                
                // Carrega materiais se disponível
                if (modelInfo.has_mtl) {{
                    const mtlLoader = new MTLLoader();
                    mtlLoader.setPath('/modelo/{job_id}/texture/');
                    materials = await new Promise((resolve, reject) => {{
                        mtlLoader.load('/modelo/{job_id}/mtl', resolve, undefined, reject);
                    }});
                    materials.preload();
                }}
                
                // Carrega modelo OBJ
                const objLoader = new OBJLoader();
                if (materials) {{
                    objLoader.setMaterials(materials);
                }}
                
                const object = await new Promise((resolve, reject) => {{
                    objLoader.load('/modelo/{job_id}/obj', resolve, undefined, reject);
                }});
                
                // Processa o objeto carregado
                let vertexCount = 0;
                let faceCount = 0;
                
                object.traverse(function(child) {{
                    if (child.isMesh) {{
                        child.castShadow = true;
                        child.receiveShadow = true;
                        
                        if (child.geometry) {{
                            vertexCount += child.geometry.attributes.position.count;
                            if (child.geometry.index) {{
                                faceCount += child.geometry.index.count / 3;
                            }}
                        }}
                        
                        // Melhora material se não há textura
                        if (!child.material.map) {{
                            child.material = new THREE.MeshLambertMaterial({{
                                color: 0x888888,
                                vertexColors: child.geometry.attributes.color ? true : false
                            }});
                        }}
                    }}
                }});
                
                // Centraliza e escala o modelo
                const box = new THREE.Box3().setFromObject(object);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                
                object.position.sub(center);
                scene.add(object);
                currentModel = object;
                
                // Posiciona câmera
                camera.position.set(maxDim, maxDim, maxDim);
                camera.lookAt(0, 0, 0);
                controls.target.set(0, 0, 0);
                controls.update();
                
                showStatus(`Modelo 3D carregado<br>Vértices: ${{vertexCount.toLocaleString()}}<br>Faces: ${{faceCount.toLocaleString()}}<br>Dimensões: ${{size.x.toFixed(2)}} x ${{size.y.toFixed(2)}} x ${{size.z.toFixed(2)}}`, 'success');
                
                // Define modo inicial como malha para modelos OBJ
                toggleRenderMode('mesh');
                
            }} catch (error) {{
                console.error('Erro ao carregar OBJ:', error);
                showStatus('Erro ao carregar modelo OBJ, tentando PLY...', 'warning');
                await loadPLYModel();
            }}
        }}
        
        // Carrega modelo PLY (fallback)
        async function loadPLYModel() {{
            const loader = new PLYLoader();
            
            try {{
                const geometry = await new Promise((resolve, reject) => {{
                    loader.load('/modelo/{job_id}', resolve, undefined, reject);
                }});
                
                if (!geometry.attributes.position || geometry.attributes.position.count === 0) {{
                    showStatus('Modelo inválido - sem vértices', 'error');
                    return;
                }}
                
                const pointCount = geometry.attributes.position.count;
                
                // Calcula bounding box
                geometry.computeBoundingBox();
                const box = geometry.boundingBox;
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                
                // Centraliza modelo
                geometry.translate(-center.x, -center.y, -center.z);
                
                // Cria material
                const material = new THREE.PointsMaterial({{
                    size: Math.max(0.01, maxDim / 1000),
                    vertexColors: geometry.attributes.color ? true : false,
                    color: geometry.attributes.color ? undefined : 0x00ff00,
                    sizeAttenuation: true
                }});
                
                // Cria mesh
                const points = new THREE.Points(geometry, material);
                scene.add(points);
                currentModel = points;
                
                // Posiciona câmera
                camera.position.set(maxDim, maxDim, maxDim);
                camera.lookAt(0, 0, 0);
                controls.target.set(0, 0, 0);
                controls.update();
                
                showStatus(`Nuvem de pontos carregada<br>Pontos: ${{pointCount.toLocaleString()}}<br>Dimensões: ${{size.x.toFixed(2)}} x ${{size.y.toFixed(2)}} x ${{size.z.toFixed(2)}}`, 'success');
                
                // Define modo inicial como pontos para modelos PLY
                toggleRenderMode('points');
                
            }} catch (error) {{
                console.error('Erro ao carregar PLY:', error);
                showStatus('Erro ao carregar modelo: ' + error.message, 'error');
            }}
        }}
        
        // Inicia carregamento
        loadModelInfo();
        
        // Loop de animação
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        
        // Resize handler
        window.addEventListener('resize', function() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/modelo/{job_id}")
def get_ply(job_id: str):
    ply_path = BASE_DIR / job_id / "model.ply"
    if not ply_path.exists():
        return JSONResponse(status_code=404, content={"error": "Modelo não encontrado"})
    return FileResponse(ply_path, media_type='application/octet-stream')

@app.get("/modelo/{job_id}/obj")
def get_obj(job_id: str):
    """Serve o modelo OBJ com texturas"""
    obj_path = BASE_DIR / job_id / "model.obj"
    if not obj_path.exists():
        return JSONResponse(status_code=404, content={"error": "Modelo OBJ não encontrado"})
    return FileResponse(obj_path, media_type='text/plain')

@app.get("/modelo/{job_id}/mtl")
def get_mtl(job_id: str):
    """Serve o arquivo MTL (material)"""
    mtl_path = BASE_DIR / job_id / "model.obj.mtl"
    if not mtl_path.exists():
        return JSONResponse(status_code=404, content={"error": "Arquivo de material não encontrado"})
    return FileResponse(mtl_path, media_type='text/plain')

@app.get("/modelo/{job_id}/texture/{filename}")
def get_texture(job_id: str, filename: str):
    """Serve arquivos de textura"""
    texture_path = BASE_DIR / job_id / filename
    if not texture_path.exists():
        return JSONResponse(status_code=404, content={"error": "Textura não encontrada"})
    
    # Determina o tipo MIME baseado na extensão
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        media_type = 'image/jpeg'
    elif filename.lower().endswith('.png'):
        media_type = 'image/png'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(texture_path, media_type=media_type)

@app.get("/modelo/{job_id}/info")
def get_model_info(job_id: str):
    """Retorna informações sobre os modelos disponíveis"""
    job_path = BASE_DIR / job_id
    
    info = {
        "has_ply": (job_path / "model.ply").exists(),
        "has_obj": (job_path / "model.obj").exists(),
        "has_mtl": (job_path / "model.obj.mtl").exists(),
        "textures": []
    }
    
    # Lista texturas disponíveis
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        info["textures"].extend([f.name for f in job_path.glob(ext)])
    
    return info

@app.get("/upload.html")
def get_upload_page():
    """Serve a página de upload"""
    try:
        with open("upload.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Página de upload não encontrada"})

@app.get("/")
def redirect_to_upload():
    """Redireciona para a página de upload"""
    return HTMLResponse(content='<script>window.location.href="/upload.html"</script>')

def extract_best_frames(video_path: str, output_dir: Path, target_frames: int = 50) -> int:
    """
    Extrai os melhores frames de um vídeo usando algoritmos de qualidade
    """
    print(f"Tentando abrir vídeo: {video_path}")
    
    # Verifica se o arquivo existe
    if not os.path.exists(video_path):
        print(f"ERRO: Arquivo de vídeo não encontrado: {video_path}")
        return 0
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Verifica se conseguiu abrir o vídeo
    if not cap.isOpened():
        print(f"ERRO: Não foi possível abrir o vídeo: {video_path}")
        return 0
    
    # Informações do vídeo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Vídeo carregado: {total_frames} frames, {fps} FPS, {width}x{height}")
    
    if total_frames == 0:
        print("ERRO: Vídeo sem frames válidos")
        cap.release()
        return 0
    
    # Intervalo inicial para extração (mais conservador)
    frame_interval = max(1, total_frames // (target_frames * 2))
    print(f"Intervalo de frames: {frame_interval}")
    
    frames_data = []
    frame_count = 0
    extracted_count = 0
    
    # Primeira passada: extrai frames candidatos
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            try:
                # Calcula métricas de qualidade
                blur_score = calculate_blur_score(frame)
                brightness_score = calculate_brightness_score(frame)
                contrast_score = calculate_contrast_score(frame)
                edge_score = calculate_edge_score(frame)
                
                # Score geral (maior é melhor) - prioriza nitidez e detalhes para texturas
                quality_score = blur_score * 0.4 + edge_score * 0.3 + contrast_score * 0.2 + brightness_score * 0.1
                
                frames_data.append({
                    'frame_number': frame_count,
                    'frame': frame.copy(),
                    'quality_score': quality_score,
                    'blur_score': blur_score
                })
                
                extracted_count += 1
                if extracted_count % 10 == 0:
                    print(f"Extraídos {extracted_count} frames candidatos...")
                    
            except Exception as e:
                print(f"Erro ao processar frame {frame_count}: {e}")
            
        frame_count += 1
    
    cap.release()
    print(f"Total de frames candidatos extraídos: {len(frames_data)}")
    
    if len(frames_data) == 0:
        print("ERRO: Nenhum frame foi extraído do vídeo")
        return 0
    
    # Filtra frames muito borrados (threshold otimizado para reconstrução 3D)
    original_count = len(frames_data)
    blur_threshold = 100 if len(frames_data) > target_frames * 2 else 50
    frames_data = [f for f in frames_data if f['blur_score'] > blur_threshold]
    print(f"Frames após filtro de blur: {len(frames_data)} (removidos {original_count - len(frames_data)})")
    
    if len(frames_data) == 0:
        print("AVISO: Todos os frames foram filtrados por blur, usando threshold mais baixo")
        cap = cv2.VideoCapture(str(video_path))
        frames_data = []
        frame_count = 0
        
        # Segunda tentativa sem filtro de blur
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames_data.append({
                    'frame_number': frame_count,
                    'frame': frame.copy(),
                    'quality_score': 100,  # score padrão
                    'blur_score': 100
                })
                
            frame_count += 1
        cap.release()
    
    # Ordena por qualidade e pega os melhores
    frames_data.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # Garante diversidade temporal
    selected_frames = select_diverse_frames(frames_data, min(target_frames, len(frames_data)))
    
    # Salva os frames selecionados
    saved_count = 0
    for i, frame_data in enumerate(selected_frames):
        try:
            filename = output_dir / f"frame_{i:04d}.jpg"
            success = cv2.imwrite(str(filename), frame_data['frame'])
            if success:
                saved_count += 1
            else:
                print(f"ERRO: Falha ao salvar frame {i}")
        except Exception as e:
            print(f"ERRO ao salvar frame {i}: {e}")
    
    print(f"Frames salvos com sucesso: {saved_count}")
    return saved_count


def calculate_blur_score(frame):
    """Calcula o score de nitidez usando Laplacian variance"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def calculate_brightness_score(frame):
    """Calcula score de brightness (penaliza muito escuro ou muito claro)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    # Score máximo em torno de 128 (meio termo)
    return 100 - abs(mean_brightness - 128) / 128 * 100


def calculate_contrast_score(frame):
    """Calcula score de contraste"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.std(gray)


def calculate_edge_score(frame):
    """Calcula score baseado na quantidade de bordas (detalhes)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges) / (frame.shape[0] * frame.shape[1])


def select_diverse_frames(frames_data, target_count):
    """Seleciona frames garantindo diversidade temporal"""
    if len(frames_data) <= target_count:
        return frames_data
    
    # Ordena por frame number para manter ordem temporal
    frames_data.sort(key=lambda x: x['frame_number'])
    
    selected = []
    step = len(frames_data) / target_count
    
    for i in range(target_count):
        idx = int(i * step)
        if idx < len(frames_data):
            selected.append(frames_data[idx])
    
    return selected
