"""
MCP4 - 视频音频合并服务（静态资源版）
用于演示的模拟服务，返回预制的合成视频
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import uuid
from pathlib import Path
from datetime import datetime

# ==================== 初始化FastAPI ====================
app = FastAPI(
    title="MCP4 - Video Audio Merge Service (Static)",
    description="合并视频和音频服务（静态资源版）",
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
VEDIOS_DIR = Path("vedios")
AUDIO_DIR = Path("audio_outputs")
OUTPUTS_DIR = Path("outputs")

# 确保目录存在
OUTPUTS_DIR.mkdir(exist_ok=True)

# 静态文件服务
if VEDIOS_DIR.exists():
    app.mount("/vedios", StaticFiles(directory="vedios"), name="vedios")
if AUDIO_DIR.exists():
    app.mount("/audio_outputs", StaticFiles(directory="audio_outputs"), name="audio_outputs")
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ==================== 数据模型 ====================

class MergeVideoAudioRequest(BaseModel):
    """合并视频音频请求模型"""
    video_urls: List[str]  # 视频片段URL列表（5个4秒视频）
    audio_url: str  # 音频URL（10秒音频）
    gpt_prompt: str  # DeepSeek提炼的核心prompt
    audio_loop: Optional[int] = 2  # 音频循环次数（默认2次变成20秒）

class MergeVideoAudioResponse(BaseModel):
    """合并视频音频响应模型"""
    status: str
    final_video_url: str  # 最终合成视频的URL

# ==================== 合并管理 ====================

class MergeManager:
    """合并管理器 - 管理视频音频合并"""
    
    def __init__(self):
        self.final_video = "6.mp4"  # 预制的合成视频
        self.sessions = {}  # 存储会话信息
    
    def get_final_video(self, session_id: Optional[str] = None) -> str:
        """获取最终合成视频"""
        # 对于演示，返回预制的合成视频（6.mp4）
        video_path = f"/vedios/{self.final_video}"
        
        # 记录会话信息
        if session_id:
            self.sessions[session_id] = {
                "final_video": self.final_video,
                "timestamp": datetime.now().isoformat()
            }
        
        return video_path
    
    def check_final_video_exists(self) -> bool:
        """检查最终视频文件是否存在"""
        video_file = VEDIOS_DIR / self.final_video
        return video_file.exists()
    
    def validate_inputs(self, video_urls: List[str], audio_url: str) -> dict:
        """验证输入的视频和音频URL"""
        validation_result = {
            "valid": True,
            "video_count": len(video_urls),
            "expected_video_count": 5,
            "audio_provided": bool(audio_url),
            "issues": []
        }
        
        # 检查视频数量
        if len(video_urls) != 5:
            validation_result["valid"] = False
            validation_result["issues"].append(f"需要5个视频片段，实际收到{len(video_urls)}个")
        
        # 检查音频
        if not audio_url:
            validation_result["valid"] = False
            validation_result["issues"].append("缺少音频URL")
        
        return validation_result

# 初始化合并管理器
merge_manager = MergeManager()

# ==================== API端点 ====================

@app.post("/mcp4/merge-video-audio", response_model=MergeVideoAudioResponse)
async def merge_video_audio(request: MergeVideoAudioRequest):
    """
    MCP4标准接口：合并视频和音频（静态实现）
    
    直接返回预制的6.mp4（1-5.mp4的合成视频）
    """
    print("\n" + "=" * 80)
    print(f"[MCP4 Static] 收到视频音频合并请求")
    print(f"[MCP4 Static] 请求时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MCP4 Static] 请求参数:")
    print(f"  - 视频片段数: {len(request.video_urls)}")
    for i, video_url in enumerate(request.video_urls, 1):
        print(f"    片段{i}: {video_url}")
    print(f"  - 音频URL: {request.audio_url}")
    print(f"  - 音频循环: {request.audio_loop}次")
    print(f"  - GPT Prompt: {request.gpt_prompt}")
    
    try:
        # 验证输入
        validation = merge_manager.validate_inputs(request.video_urls, request.audio_url)
        if not validation["valid"]:
            print(f"[MCP4 Static] 输入验证失败: {validation['issues']}")
        
        # 模拟处理延迟（4-5秒）
        print(f"[MCP4 Static] 开始合成处理...")
        await asyncio.sleep(4.5)  # 延迟4.5秒，模拟合成过程
        
        # 检查最终视频是否存在
        if not merge_manager.check_final_video_exists():
            print(f"[MCP4 Static] 警告: 最终视频文件不存在")
        
        # 直接返回预制的合成视频
        final_video_url = merge_manager.get_final_video()
        
        print(f"[MCP4 Static] 返回合成视频: {final_video_url}")
        print(f"[MCP4 Static] 视频规格: 20秒, 1024x768, 带音频")
        print(f"[MCP4 Static] 活跃会话数: {len(merge_manager.sessions)}")
        print("=" * 80 + "\n")
        
        # 构建响应
        return MergeVideoAudioResponse(
            status="success",
            final_video_url=final_video_url
        )
        
    except Exception as e:
        error_msg = f"合并失败: {str(e)}"
        print(f"[MCP4 Static] 错误: {error_msg}")
        print("=" * 80 + "\n")
        
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
async def root():
    """服务信息"""
    return {
        "service": "MCP4 - Video Audio Merge (Static)",
        "description": "合并视频和音频服务（静态资源版）",
        "status": "running",
        "final_video": merge_manager.final_video,
        "final_video_exists": merge_manager.check_final_video_exists(),
        "endpoints": {
            "merge_video_audio": "/mcp4/merge-video-audio",
            "health": "/health",
            "sessions": "/sessions",
            "test_video": "/test-video"
        }
    }

@app.get("/health")
async def health():
    """健康检查"""
    # 检查文件状态
    video_files = []
    audio_files = []
    
    if VEDIOS_DIR.exists():
        for video_file in VEDIOS_DIR.glob("*.mp4"):
            video_files.append({
                "name": video_file.name,
                "size": video_file.stat().st_size,
                "is_final": video_file.name == merge_manager.final_video
            })
    
    if AUDIO_DIR.exists():
        for audio_file in AUDIO_DIR.glob("*.mp3"):
            audio_files.append({
                "name": audio_file.name,
                "size": audio_file.stat().st_size
            })
    
    return {
        "status": "healthy",
        "vedios_dir": str(VEDIOS_DIR),
        "audio_dir": str(AUDIO_DIR),
        "outputs_dir": str(OUTPUTS_DIR),
        "final_video": merge_manager.final_video,
        "final_video_exists": merge_manager.check_final_video_exists(),
        "available_videos": video_files,
        "available_audio": audio_files,
        "active_sessions": len(merge_manager.sessions)
    }

@app.get("/sessions")
async def get_sessions():
    """获取所有会话信息"""
    return {
        "sessions": merge_manager.sessions,
        "total_sessions": len(merge_manager.sessions)
    }

# ==================== 测试端点 ====================

@app.post("/test")
async def test_merge():
    """测试合并功能"""
    test_request = MergeVideoAudioRequest(
        video_urls=[
            "/vedios/1.mp4",
            "/vedios/2.mp4",
            "/vedios/3.mp4",
            "/vedios/4.mp4",
            "/vedios/5.mp4"
        ],
        audio_url="/audio_outputs/mock_audio_turn1.mp3",
        gpt_prompt="A test merge operation",
        audio_loop=2
    )
    
    return await merge_video_audio(test_request)

@app.get("/test-video")
async def test_video_playback():
    """测试视频播放页面"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP4 视频测试</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #667eea;
                padding-bottom: 15px;
                margin-bottom: 20px;
            }}
            video {{
                width: 100%;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                margin: 20px 0;
            }}
            .info {{
                background: linear-gradient(135deg, #e8eaf6 0%, #f3e5f5 100%);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            .info p {{
                margin: 10px 0;
                color: #555;
            }}
            .info strong {{
                color: #333;
                display: inline-block;
                width: 120px;
            }}
            .status {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            .status.exists {{
                background: #4caf50;
                color: white;
            }}
            .status.missing {{
                background: #f44336;
                color: white;
            }}
            .workflow {{
                background: #f5f5f5;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
            }}
            .workflow h2 {{
                color: #667eea;
                font-size: 18px;
                margin-bottom: 15px;
            }}
            .workflow ul {{
                list-style: none;
                padding: 0;
            }}
            .workflow li {{
                padding: 8px 0;
                border-bottom: 1px solid #ddd;
            }}
            .workflow li:last-child {{
                border-bottom: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 MCP4 最终视频测试</h1>
            
            <div class="info">
                <p><strong>视频文件:</strong> {merge_manager.final_video}</p>
                <p><strong>视频时长:</strong> 20秒</p>
                <p><strong>分辨率:</strong> 1024×768</p>
                <p><strong>文件状态:</strong> 
                    <span class="status {'exists' if merge_manager.check_final_video_exists() else 'missing'}">
                        {'存在' if merge_manager.check_final_video_exists() else '不存在'}
                    </span>
                </p>
            </div>
            
            <video controls>
                <source src="/vedios/{merge_manager.final_video}" type="video/mp4">
                您的浏览器不支持视频播放。
            </video>
            
            <div class="workflow">
                <h2>工作流说明</h2>
                <ul>
                    <li>📝 输入: 5个4秒视频片段（1.mp4 - 5.mp4）</li>
                    <li>🎵 输入: 1个10秒音频（循环2次）</li>
                    <li>🎬 输出: 1个20秒完整视频（6.mp4）</li>
                    <li>⚙️ 处理: 静态服务直接返回预制视频</li>
                </ul>
            </div>
            
            <p style="text-align: center; color: #888; margin-top: 30px;">
                这是用于演示的预制合成视频文件
            </p>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# ==================== 启动服务 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 80)
    print("MCP4 - 视频音频合并服务（静态资源版）")
    print("=" * 80)
    print(f"视频目录: {VEDIOS_DIR}")
    print(f"音频目录: {AUDIO_DIR}")
    print(f"输出目录: {OUTPUTS_DIR}")
    print(f"最终视频: {merge_manager.final_video}")
    
    # 检查最终视频文件
    if merge_manager.check_final_video_exists():
        video_file = VEDIOS_DIR / merge_manager.final_video
        print(f"最终视频存在: {video_file}")
        print(f"文件大小: {video_file.stat().st_size} bytes")
    else:
        print("警告: 最终视频文件不存在！")
        print(f"请确保文件 '{merge_manager.final_video}' 在 '{VEDIOS_DIR}' 目录中")
    
    # 检查源视频片段
    print("\n源视频片段检查:")
    for i in range(1, 6):
        video_path = VEDIOS_DIR / f"{i}.mp4"
        if video_path.exists():
            print(f"  ✓ {i}.mp4 存在")
        else:
            print(f"  ✗ {i}.mp4 缺失")
    
    print("\n端点:")
    print("  POST /mcp4/merge-video-audio - 合并视频音频")
    print("  GET  /health                 - 健康检查")
    print("  GET  /sessions               - 查看会话信息")
    print("  POST /test                   - 测试合并")
    print("  GET  /test-video             - 视频播放测试页面")
    print("=" * 80 + "\n")
    
    # 启动服务，监听8004端口，关闭自动重载
    uvicorn.run(app, host="0.0.0.0", port=8004, reload=False)