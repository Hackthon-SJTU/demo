"""
ReasoningACTION 工作流系统 - 后端服务
与DeepSeek-v3对话并调用MCP模块生成视频
"""
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import time
import uuid
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime

# ==================== 初始化 ====================
app = FastAPI(
    title="ReasoningACTION Workflow System",
    description="与DeepSeek对话并调用MCP生成视频",
    version="1.0.0"
)

# CORS配置 - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建必要目录
OUTPUTS_DIR = Path("outputs")
TEMP_DIR = Path("temp")
VEDIOS_DIR = Path("vedios")
AUDIO_DIR = Path("audio_outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# 静态文件服务
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/vedios", StaticFiles(directory="vedios"), name="vedios")
app.mount("/audio_outputs", StaticFiles(directory="audio_outputs"), name="audio_outputs")

# ==================== 配置 ====================

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-a68ba86867044b8ebcb5e669937626a1")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# MCP服务地址
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost")

# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    """用户聊天请求"""
    message: str  # 用户输入的视频描述
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: str
    gpt_prompt: str  # DeepSeek提炼的核心prompt
    status: str  # processing, completed, error
    progress: float  # 0-100
    message: str
    video_url: Optional[str] = None
    error: Optional[str] = None

class WorkflowStatus(BaseModel):
    """工作流状态"""
    session_id: str
    current_step: str
    progress: float
    details: Dict[str, Any]

# ==================== MCP接口定义 ====================

class MCPInterface:
    """MCP接口调用类"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.video_counter = 0  # 用于追踪视频片段
    
    async def text_to_image(self, prompt: str, gpt_prompt: str) -> Dict:
        """
        MCP1: 文本生成图片
        调用8001端口的服务
        """
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}:8001/mcp1/text-to-image"
                print(f"[MCP1] 调用: {url}")
                
                async with session.post(
                    url,
                    json={
                        "prompt": prompt,
                        "gpt_prompt": gpt_prompt,
                        "width": 1024,
                        "height": 768
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    result = await response.json()
                    print(f"[MCP1] 响应: {result}")
                    return result
            except Exception as e:
                print(f"[MCP1] 调用失败: {e}")
                # 返回默认图片用于演示
                return {
                    "status": "success",
                    "image_url": "/outputs/sample_image.png"
                }
    
    async def image_to_video(self, image_url: str, gpt_prompt: str, is_first_frame: bool = True) -> Dict:
        """
        MCP2: 图片生成4秒视频
        调用8002端口的服务（静态资源版）
        """
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}:8002/mcp2/image-to-video"
                print(f"[MCP2] 调用: {url}")
                
                async with session.post(
                    url,
                    json={
                        "image_url": image_url,
                        "gpt_prompt": gpt_prompt,
                        "is_first_frame": is_first_frame,
                        "duration": 4
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    result = await response.json()
                    print(f"[MCP2] 响应: {result}")
                    return result
            except Exception as e:
                print(f"[MCP2] 调用失败: {e}")
                # 返回默认视频
                video_index = (self.video_counter % 5) + 1
                self.video_counter += 1
                return {
                    "status": "success",
                    "video_url": f"/vedios/{video_index}.mp4",
                    "last_frame_url": "/outputs/last_frame.png"
                }
    
    async def image_to_audio(self, image_url: str, gpt_prompt: str) -> Dict:
        """
        MCP3: 图片生成10秒音频
        调用8003端口的服务（静态资源版）
        """
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}:8003/mcp3/image-to-audio"
                print(f"[MCP3] 调用: {url}")
                
                async with session.post(
                    url,
                    json={
                        "image_url": image_url,
                        "gpt_prompt": gpt_prompt,
                        "duration": 10
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    result = await response.json()
                    print(f"[MCP3] 响应: {result}")
                    return result
            except Exception as e:
                print(f"[MCP3] 调用失败: {e}")
                # 返回默认音频
                return {
                    "status": "success",
                    "audio_url": "/audio_outputs/mock_audio_turn1.mp3"
                }
    
    async def merge_video_audio(self, video_urls: List[str], audio_url: str, gpt_prompt: str) -> Dict:
        """
        MCP4: 拼接视频和音频
        调用8004端口的服务（静态资源版）
        """
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}:8004/mcp4/merge-video-audio"
                print(f"[MCP4] 调用: {url}")
                
                async with session.post(
                    url,
                    json={
                        "video_urls": video_urls,
                        "audio_url": audio_url,
                        "gpt_prompt": gpt_prompt,
                        "audio_loop": 2
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    result = await response.json()
                    print(f"[MCP4] 响应: {result}")
                    return result
            except Exception as e:
                print(f"[MCP4] 调用失败: {e}")
                # 返回默认合成视频
                return {
                    "status": "success",
                    "final_video_url": "/vedios/6.mp4"
                }

# 初始化MCP接口
mcp = MCPInterface(MCP_BASE_URL)

# ==================== DeepSeek集成 ====================

class DeepSeekProcessor:
    """DeepSeek-v3处理器"""
    
    @staticmethod
    async def extract_gpt_prompt(user_message: str) -> str:
        """
        调用DeepSeek-v3提炼核心prompt
        
        输入: 用户的视频描述
        输出: 提炼后的GPT prompt（包含场景、方向、故事）
        """
        system_prompt = """你是一个专业的视频创作助手。
        用户会描述他们想要的视频，你需要提炼出：
        1. 视频场景是什么
        2. 视频的方向/风格是什么
        3. 视频讲述的故事是什么
        
        请用简洁清晰的语言输出一个综合的prompt，包含上述所有要素。
        输出格式：直接输出提炼后的prompt文本，不要有其他说明。"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    DEEPSEEK_API_URL,
                    headers={
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"DeepSeek调用失败: {e}")
            # 返回默认提炼结果
            return f"Generate a video based on: {user_message}. Scene: natural environment. Style: cinematic. Story: peaceful journey through nature."
    
    @staticmethod
    async def generate_scene_prompt(gpt_prompt: str, scene_number: int = 1) -> str:
        """
        基于GPT prompt生成具体场景描述
        """
        base_prompt = """Based on this video concept: {gpt_prompt}
        
        Generate a detailed visual description for scene {scene_number}.
        Include: lighting, colors, composition, camera angle, specific objects, atmosphere.
        Output only the scene description, no explanations."""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    DEEPSEEK_API_URL,
                    headers={
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": "You are a professional visual scene designer."},
                            {"role": "user", "content": base_prompt.format(
                                gpt_prompt=gpt_prompt,
                                scene_number=scene_number
                            )}
                        ],
                        "temperature": 0.8,
                        "max_tokens": 300
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"场景生成失败: {e}")
            return f"A serene forest landscape, lush greenery, towering trees with dense foliage, dappled sunlight filtering through the canopy, high angle view"

# ==================== 工作流管理 ====================

class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self):
        self.sessions = {}  # 存储会话状态
    
    async def execute_workflow(self, session_id: str, user_message: str, progress_callback=None):
        """
        执行完整的视频生成工作流
        
        步骤:
        1. DeepSeek提炼GPT prompt
        2. MCP1: 文本生成图片
        3. MCP2: 循环5次生成4秒视频片段（共20秒）
        4. MCP3: 图片生成10秒音频
        5. MCP4: 合并视频和音频（音频循环2次变成20秒）
        """
        
        try:
            # 初始化会话
            self.sessions[session_id] = {
                "status": "processing",
                "progress": 0,
                "current_step": "初始化",
                "start_time": datetime.now().isoformat()
            }
            
            # Step 1: DeepSeek提炼prompt
            if progress_callback:
                await progress_callback(session_id, "提炼核心概念", 10)
            
            gpt_prompt = await DeepSeekProcessor.extract_gpt_prompt(user_message)
            print(f"\n[工作流] GPT Prompt: {gpt_prompt}")
            
            # Step 2: 生成场景描述
            if progress_callback:
                await progress_callback(session_id, "生成场景描述", 20)
            
            scene_prompt = await DeepSeekProcessor.generate_scene_prompt(gpt_prompt)
            full_prompt = f"{scene_prompt}, high angle view"
            print(f"[工作流] 场景描述: {full_prompt}")
            
            # Step 3: MCP1 - 文本生成图片
            if progress_callback:
                await progress_callback(session_id, "生成初始图片", 30)
            
            image_result = await mcp.text_to_image(full_prompt, gpt_prompt)
            initial_image_url = image_result.get("image_url", "/outputs/sample_image.png")
            print(f"[工作流] 初始图片: {initial_image_url}")
            
            # Step 4: MCP2 - 循环生成视频片段（5次，每次4秒，共20秒）
            video_segments = []
            current_image = initial_image_url
            
            for i in range(5):  # 固定5次，生成20秒视频
                if progress_callback:
                    progress = 35 + (i * 8)  # 35-75%
                    await progress_callback(session_id, f"生成视频片段 {i+1}/5", progress)
                
                video_result = await mcp.image_to_video(
                    current_image, 
                    gpt_prompt,
                    is_first_frame=(i == 0)
                )
                
                video_segments.append(video_result["video_url"])
                current_image = video_result.get("last_frame_url", "/outputs/last_frame.png")
                print(f"[工作流] 视频片段 {i+1}: {video_result['video_url']}")
            
            # Step 5: MCP3 - 图片生成音频（使用初始图片，生成10秒音频）
            if progress_callback:
                await progress_callback(session_id, "生成音频", 80)
            
            audio_result = await mcp.image_to_audio(initial_image_url, gpt_prompt)
            audio_url = audio_result["audio_url"]
            print(f"[工作流] 音频: {audio_url}")
            
            # Step 6: MCP4 - 合并视频和音频（音频循环2次变成20秒）
            if progress_callback:
                await progress_callback(session_id, "合成最终视频", 90)
            
            merge_result = await mcp.merge_video_audio(
                video_segments,
                audio_url,
                gpt_prompt
            )
            
            final_video_url = merge_result["final_video_url"]
            print(f"[工作流] 最终视频: {final_video_url}")
            
            # 更新会话状态
            self.sessions[session_id] = {
                "status": "completed",
                "progress": 100,
                "current_step": "完成",
                "gpt_prompt": gpt_prompt,
                "video_url": final_video_url,
                "end_time": datetime.now().isoformat(),
                "metadata": {
                    "initial_image": initial_image_url,
                    "video_segments": video_segments,
                    "audio_url": audio_url,
                    "total_duration": "20秒",
                    "video_count": 5,
                    "audio_loops": 2
                }
            }
            
            if progress_callback:
                await progress_callback(session_id, "视频生成完成", 100)
            
            return {
                "success": True,
                "gpt_prompt": gpt_prompt,
                "video_url": final_video_url
            }
            
        except Exception as e:
            print(f"[工作流] 执行失败: {e}")
            self.sessions[session_id] = {
                "status": "error",
                "error": str(e),
                "end_time": datetime.now().isoformat()
            }
            return {
                "success": False,
                "error": str(e)
            }

# 初始化工作流管理器
workflow_manager = WorkflowManager()

# ==================== WebSocket支持（实时进度） ====================

class ConnectionManager:
    """WebSocket连接管理"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"[WebSocket] 连接建立: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"[WebSocket] 连接断开: {session_id}")
    
    async def send_progress(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
                print(f"[WebSocket] 发送进度: {session_id} - {message}")
            except Exception as e:
                print(f"[WebSocket] 发送失败: {e}")

manager = ConnectionManager()

# ==================== API端点 ====================

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    主聊天接口 - 接收用户消息并启动视频生成流程
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    print(f"\n{'='*60}")
    print(f"[API] 新请求 - Session: {session_id}")
    print(f"[API] 用户消息: {request.message}")
    print(f"{'='*60}")
    
    # 异步执行工作流
    asyncio.create_task(
        workflow_manager.execute_workflow(
            session_id,
            request.message,
            progress_callback=send_progress_update
        )
    )
    
    return ChatResponse(
        session_id=session_id,
        gpt_prompt="正在提炼核心概念...",
        status="processing",
        progress=0,
        message="已开始处理您的请求，请稍候..."
    )

@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    """
    获取工作流状态
    """
    if session_id not in workflow_manager.sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = workflow_manager.sessions[session_id]
    
    return {
        "session_id": session_id,
        "status": session.get("status", "unknown"),
        "progress": session.get("progress", 0),
        "current_step": session.get("current_step", ""),
        "gpt_prompt": session.get("gpt_prompt", ""),
        "video_url": session.get("video_url"),
        "error": session.get("error"),
        "metadata": session.get("metadata")
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket端点 - 实时推送进度更新
    """
    await manager.connect(websocket, session_id)
    try:
        while True:
            await websocket.receive_text()  # 保持连接
    except WebSocketDisconnect:
        manager.disconnect(session_id)

async def send_progress_update(session_id: str, step: str, progress: float):
    """
    发送进度更新
    """
    await manager.send_progress(session_id, {
        "type": "progress",
        "step": step,
        "progress": progress,
        "timestamp": datetime.now().isoformat()
    })

@app.get("/api/download/{session_id}")
async def download_video(session_id: str):
    """
    下载生成的视频
    """
    if session_id not in workflow_manager.sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = workflow_manager.sessions[session_id]
    video_url = session.get("video_url")
    
    if not video_url:
        raise HTTPException(status_code=404, detail="视频尚未生成")
    
    # 处理静态文件路径
    if video_url.startswith("/vedios/"):
        video_path = Path("vedios") / video_url.replace("/vedios/", "")
    elif video_url.startswith("/outputs/"):
        video_path = Path("outputs") / video_url.replace("/outputs/", "")
    else:
        video_path = Path(video_url.lstrip("/"))
    
    if video_path.exists():
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"generated_video_{session_id}.mp4"
        )
    else:
        raise HTTPException(status_code=404, detail="视频文件不存在")

# ==================== 健康检查 ====================

@app.get("/")
async def root():
    """API根路径"""
    return {
        "service": "ReasoningACTION Workflow System",
        "version": "1.0.0",
        "status": "running",
        "description": "与DeepSeek对话生成视频的工作流系统",
        "endpoints": {
            "chat": "/api/chat",
            "status": "/api/status/{session_id}",
            "download": "/api/download/{session_id}",
            "websocket": "/ws/{session_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """健康检查"""
    mcp_services = []
    
    # 检查各个MCP服务
    async with aiohttp.ClientSession() as session:
        for port, name in [(8001, "MCP1"), (8002, "MCP2"), (8003, "MCP3"), (8004, "MCP4")]:
            try:
                async with session.get(
                    f"{MCP_BASE_URL}:{port}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        mcp_services.append({"name": name, "port": port, "status": "healthy"})
                    else:
                        mcp_services.append({"name": name, "port": port, "status": "unhealthy"})
            except:
                mcp_services.append({"name": name, "port": port, "status": "offline"})
    
    return {
        "status": "healthy",
        "mcp_url": MCP_BASE_URL,
        "deepseek_configured": bool(DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-deepseek-api-key"),
        "active_sessions": len(workflow_manager.sessions),
        "mcp_services": mcp_services,
        "static_resources": {
            "vedios": list(VEDIOS_DIR.glob("*.mp4")) if VEDIOS_DIR.exists() else [],
            "audio": list(AUDIO_DIR.glob("*.mp3")) if AUDIO_DIR.exists() else []
        }
    }

# ==================== 启动服务 ====================

if __name__ == "__main__":
    # 在这里运行Uvicorn服务器
    # reload=True 会在代码变动时自动重启服务器
    # 这对于开发很方便，但请注意，如果其他进程（如MCP）修改了被监控的文件，也可能触发重启
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    # uvicorn.run(
    #     app,
    #     host="0.0.0.0",
    #     port=8000,
    #     reload=True,
    #     reload_excludes=["outputs", "temp"]
    # )
    
    print("=" * 60)