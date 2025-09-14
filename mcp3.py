"""
MCP3 - 图片生成音频服务
服务，返回预录制的音频文件
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import asyncio
from pathlib import Path
from datetime import datetime

# ==================== 初始化FastAPI ====================
app = FastAPI(
    title="MCP3 - Image to Audio Service",
    description="图片生成10秒音频服务",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置目录路径
AUDIO_DIR = Path("audio_outputs")
OUTPUTS_DIR = Path("outputs")

# 确保目录存在
AUDIO_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# 静态文件服务
if AUDIO_DIR.exists():
    app.mount("/audio_outputs", StaticFiles(directory="audio_outputs"), name="audio_outputs")
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ==================== 数据模型 ====================

class ImageToAudioRequest(BaseModel):
    """图片生成音频请求模型"""
    image_url: str  # 输入图片的URL或路径
    gpt_prompt: str  # DeepSeek提炼的核心prompt
    duration: Optional[int] = 10  # 音频时长（固定10秒）

class ImageToAudioResponse(BaseModel):
    """图片生成音频响应模型"""
    status: str
    audio_url: str  # 生成的音频URL

# ==================== 音频管理 ====================

class AudioManager:
    """音频管理器 - 管理音频文件的分配"""
    
    def __init__(self):
        self.default_audio = "mock_audio_turn1.mp3"  # 默认音频文件
        self.sessions = {}  # 存储会话信息
    
    def get_audio_for_session(self, session_id: Optional[str] = None) -> str:
        """根据会话获取音频"""
        # 对于演示，总是返回同一个预录制的音频
        audio_path = f"/audio_outputs/{self.default_audio}"
        
        # 记录会话信息
        if session_id:
            self.sessions[session_id] = {
                "audio": self.default_audio,
                "timestamp": datetime.now().isoformat()
            }
        
        return audio_path
    
    def check_audio_exists(self) -> bool:
        """检查默认音频文件是否存在"""
        audio_file = AUDIO_DIR / self.default_audio
        return audio_file.exists()

# 初始化音频管理器
audio_manager = AudioManager()

# ==================== API端点 ====================

@app.post("/mcp3/image-to-audio", response_model=ImageToAudioResponse)
async def image_to_audio(request: ImageToAudioRequest):
    """
    MCP3标准接口：图片生成10秒音频
    """
    print("\n" + "=" * 80)
    print(f"[MCP3 Static] 收到图片转音频请求")
    print(f"[MCP3 Static] 请求时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MCP3 Static] 请求参数:")
    print(f"  - image_url: {request.image_url}")
    print(f"  - gpt_prompt: {request.gpt_prompt}")
    print(f"  - duration: {request.duration}秒")
    
    try:
        # 检查音频文件是否存在
        if not audio_manager.check_audio_exists():
            print(f"[MCP3 Static] 警告: 默认音频文件不存在")
        
        # 模拟处理延迟（2-3秒）
        print(f"[MCP3 Static] 正在生成音频...")
        await asyncio.sleep(2.5)  # 延迟2.5秒，模拟音频生成
        
        # 获取音频文件路径
        audio_url = audio_manager.get_audio_for_session()
        
        print(f"[MCP3 Static] 返回音频: {audio_url}")
        print(f"[MCP3 Static] 活跃会话数: {len(audio_manager.sessions)}")
        print("=" * 80 + "\n")
        
        # 构建响应
        return ImageToAudioResponse(
            status="success",
            audio_url=audio_url
        )
        
    except Exception as e:
        error_msg = f"生成失败: {str(e)}"
        print(f"[MCP3 Static] 错误: {error_msg}")
        print("=" * 80 + "\n")
        
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
async def root():
    """服务信息"""
    return {
        "service": "MCP3 - Image to Audio (Static)",
        "description": "图片生成10秒音频服务",
        "status": "running",
        "default_audio": audio_manager.default_audio,
        "audio_exists": audio_manager.check_audio_exists(),
        "endpoints": {
            "image_to_audio": "/mcp3/image-to-audio",
            "health": "/health",
            "sessions": "/sessions"
        }
    }

@app.get("/health")
async def health():
    """健康检查"""
    # 检查音频文件状态
    audio_files = []
    if AUDIO_DIR.exists():
        for audio_file in AUDIO_DIR.glob("*.mp3"):
            audio_files.append({
                "name": audio_file.name,
                "size": audio_file.stat().st_size,
                "path": str(audio_file)
            })
    
    return {
        "status": "healthy",
        "audio_dir": str(AUDIO_DIR),
        "outputs_dir": str(OUTPUTS_DIR),
        "default_audio": audio_manager.default_audio,
        "default_audio_exists": audio_manager.check_audio_exists(),
        "available_audio_files": audio_files,
        "active_sessions": len(audio_manager.sessions)
    }

@app.get("/sessions")
async def get_sessions():
    """获取所有会话信息"""
    return {
        "sessions": audio_manager.sessions,
        "total_sessions": len(audio_manager.sessions)
    }

# ==================== 测试端点 ====================

@app.post("/test")
async def test_generation():
    """测试生成功能"""
    test_request = ImageToAudioRequest(
        image_url="/outputs/test_image.png",
        gpt_prompt="A peaceful forest scene with gentle music",
        duration=10
    )
    
    return await image_to_audio(test_request)

@app.get("/test-audio")
async def test_audio_playback():
    """测试音频播放页面"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP3 音频测试</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
            }}
            audio {{
                width: 100%;
                margin: 20px 0;
            }}
            .info {{
                background: #e8f5e9;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MCP3 音频测试</h1>
            <div class="info">
                <p><strong>默认音频文件:</strong> {audio_manager.default_audio}</p>
                <p><strong>音频时长:</strong> 10秒</p>
                <p><strong>文件状态:</strong> {'存在' if audio_manager.check_audio_exists() else '不存在'}</p>
            </div>
            <audio controls>
                <source src="/audio_outputs/{audio_manager.default_audio}" type="audio/mpeg">
                您的浏览器不支持音频播放。
            </audio>
            <p>这是用于演示的预录制音频文件。</p>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# ==================== 启动服务 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 80)
    print("MCP3 - 图片生成音频服务")
    print("=" * 80)
    print(f"音频目录: {AUDIO_DIR}")
    print(f"输出目录: {OUTPUTS_DIR}")
    print(f"默认音频: {audio_manager.default_audio}")
    
    # 检查音频文件
    if audio_manager.check_audio_exists():
        audio_file = AUDIO_DIR / audio_manager.default_audio
        print(f"音频文件存在: {audio_file}")
        print(f"文件大小: {audio_file.stat().st_size} bytes")
    else:
        print("警告: 默认音频文件不存在！")
        print(f"请确保文件 '{audio_manager.default_audio}' 在 '{AUDIO_DIR}' 目录中")
    
    print("\n端点:")
    print("  POST /mcp3/image-to-audio - 图片生成音频")
    print("  GET  /health              - 健康检查")
    print("  GET  /sessions            - 查看会话信息")
    print("  POST /test                - 测试生成")
    print("  GET  /test-audio          - 音频播放测试页面")
    print("=" * 80 + "\n")
    
    # 启动服务，监听8003端口，关闭自动重载
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=False)