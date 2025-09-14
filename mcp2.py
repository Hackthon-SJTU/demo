"""
MCP2 - 图片生成视频服务
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import asyncio
from pathlib import Path
from datetime import datetime

# ==================== 初始化FastAPI ====================
app = FastAPI(
    title="MCP2 - Image to Video Service (Static)",
    description="图片生成4秒视频服务",
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
OUTPUTS_DIR = Path("outputs")

# 确保目录存在
OUTPUTS_DIR.mkdir(exist_ok=True)

# 静态文件服务
if VEDIOS_DIR.exists():
    app.mount("/vedios", StaticFiles(directory="vedios"), name="vedios")
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ==================== 数据模型 ====================

class ImageToVideoRequest(BaseModel):
    """图片生成视频请求模型"""
    image_url: str  # 输入图片的URL或路径
    gpt_prompt: str  # DeepSeek提炼的核心prompt
    is_first_frame: Optional[bool] = True  # 是否作为首帧
    duration: Optional[int] = 4  # 视频时长（固定4秒）

class ImageToVideoResponse(BaseModel):
    """图片生成视频响应模型"""
    status: str
    video_url: str  # 生成的视频URL
    last_frame_url: str  # 最后一帧的URL

# ==================== 全局状态 ====================

class VideoManager:
    """视频管理器 - 管理视频片段的分配"""
    
    def __init__(self):
        self.counter = 0  # 计数器，用于循环分配视频
        self.sessions = {}  # 存储会话状态
    
    def get_next_video(self, session_id: Optional[str] = None) -> str:
        """获取下一个视频片段"""
        # 使用1-5.mp4循环
        video_index = (self.counter % 5) + 1
        self.counter += 1
        
        # 记录会话信息
        if session_id:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append(video_index)
        
        return f"/vedios/{video_index}.mp4"
    
    def reset_counter(self):
        """重置计数器"""
        self.counter = 0

# 初始化视频管理器
video_manager = VideoManager()

# ==================== API端点 ====================

@app.post("/mcp2/image-to-video", response_model=ImageToVideoResponse)
async def image_to_video(request: ImageToVideoRequest):
    """
    MCP2标准接口：图片生成4秒视频
    
    模拟图片生成视频的过程，实际返回预录制的视频文件
    """
    print("\n" + "=" * 80)
    print(f"[MCP2 Static] 收到图片转视频请求")
    print(f"[MCP2 Static] 请求时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MCP2 Static] 请求参数:")
    print(f"  - image_url: {request.image_url}")
    print(f"  - gpt_prompt: {request.gpt_prompt}")
    print(f"  - is_first_frame: {request.is_first_frame}")
    print(f"  - duration: {request.duration}秒")
    
    try:
        # 模拟处理延迟（3-4秒）
        print(f"[MCP2 Static] 正在处理视频生成...")
        await asyncio.sleep(3.5)  # 延迟3.5秒，模拟视频生成过程
        
        # 获取下一个视频片段
        video_url = video_manager.get_next_video()
        
        # 静态的最后一帧URL（实际应用中应该从视频中提取）
        last_frame_url = "/outputs/last_frame.png"
        
        print(f"[MCP2 Static] 分配视频: {video_url}")
        print(f"[MCP2 Static] 当前计数: {video_manager.counter}")
        print("=" * 80 + "\n")
        
        # 构建响应
        return ImageToVideoResponse(
            status="success",
            video_url=video_url,
            last_frame_url=last_frame_url
        )
        
    except Exception as e:
        error_msg = f"生成失败: {str(e)}"
        print(f"[MCP2 Static] 错误: {error_msg}")
        print("=" * 80 + "\n")
        
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
async def root():
    """服务信息"""
    return {
        "service": "MCP2 - Image to Video (Static)",
        "description": "图片生成4秒视频服务",
        "status": "running",
        "available_videos": 5,
        "current_counter": video_manager.counter,
        "endpoints": {
            "image_to_video": "/mcp2/image-to-video",
            "health": "/health",
            "reset": "/reset"
        }
    }

@app.get("/health")
async def health():
    """健康检查"""
    # 检查视频文件是否存在
    available_videos = []
    if VEDIOS_DIR.exists():
        for i in range(1, 6):
            video_path = VEDIOS_DIR / f"{i}.mp4"
            if video_path.exists():
                available_videos.append(f"{i}.mp4")
    
    return {
        "status": "healthy",
        "vedios_dir": str(VEDIOS_DIR),
        "outputs_dir": str(OUTPUTS_DIR),
        "available_videos": available_videos,
        "video_count": len(available_videos),
        "current_counter": video_manager.counter,
        "active_sessions": len(video_manager.sessions)
    }

@app.post("/reset")
async def reset_counter():
    """重置视频计数器"""
    video_manager.reset_counter()
    return {
        "status": "success",
        "message": "计数器已重置",
        "current_counter": video_manager.counter
    }

@app.get("/sessions")
async def get_sessions():
    """获取所有会话信息"""
    return {
        "sessions": video_manager.sessions,
        "total_sessions": len(video_manager.sessions),
        "current_counter": video_manager.counter
    }

# ==================== 测试端点 ====================

@app.post("/test")
async def test_generation():
    """测试生成功能"""
    test_request = ImageToVideoRequest(
        image_url="/outputs/test_image.png",
        gpt_prompt="A test video generation prompt",
        is_first_frame=True,
        duration=4
    )
    
    return await image_to_video(test_request)

# ==================== 启动服务 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 80)
    print("MCP2 - 图片生成视频服务")
    print("=" * 80)
    print(f"视频目录: {VEDIOS_DIR}")
    print(f"输出目录: {OUTPUTS_DIR}")
    
    # 检查可用视频
    if VEDIOS_DIR.exists():
        videos = list(VEDIOS_DIR.glob("*.mp4"))
        print(f"可用视频: {len(videos)}个")
        for video in sorted(videos)[:5]:
            print(f"  - {video.name}")
    else:
        print("警告: vedios目录不存在！")
    
    print("\n端点:")
    print("  POST /mcp2/image-to-video - 图片生成视频")
    print("  GET  /health              - 健康检查")
    print("  POST /reset               - 重置计数器")
    print("  GET  /sessions            - 查看会话信息")
    print("  POST /test                - 测试生成")
    print("=" * 80 + "\n")
    
    # 启动服务，监听8002端口
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)