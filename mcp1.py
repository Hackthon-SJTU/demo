"""
MCP1 - 文本生成图片服务（静态资源版）
用于演示的模拟服务，返回预设的图片
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import json
import os
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
import shutil

# ==================== 初始化FastAPI ====================
app = FastAPI(
    title="MCP1 - Text to Image Service (Static)",
    description="文本生成图片服务（静态资源版）",
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

# 创建必要的目录
OUTPUTS_DIR = Path("outputs")
TEMP_DIR = Path("temp")
STATIC_IMAGES_DIR = Path("static_images")
OUTPUTS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
STATIC_IMAGES_DIR.mkdir(exist_ok=True)

# 静态文件服务
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/temp", StaticFiles(directory="temp"), name="temp")
if STATIC_IMAGES_DIR.exists():
    app.mount("/static_images", StaticFiles(directory="static_images"), name="static_images")

# ==================== 数据模型 ====================

class TextToImageRequest(BaseModel):
    """文生图请求模型"""
    prompt: str  # 场景描述
    gpt_prompt: str  # DeepSeek提炼的核心prompt
    width: Optional[int] = 1024
    height: Optional[int] = 768

class TextToImageResponse(BaseModel):
    """文生图响应模型"""
    status: str
    image_url: str
    image_data: Optional[str] = None  # base64编码（可选）

# ==================== 图片管理 ====================

class ImageManager:
    """图片管理器 - 管理静态图片资源"""
    
    def __init__(self):
        self.default_images = [
            "forest_scene_1.jpg",
            "forest_scene_2.jpg",
            "nature_landscape.jpg"
        ]
        self.counter = 0
        self.sessions = {}
        
        # 创建默认图片（如果不存在）
        self.create_default_images()
    
    def create_default_images(self):
        """创建默认的演示图片"""
        # 创建一个简单的演示图片（纯色图片作为占位符）
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            for i, filename in enumerate(self.default_images):
                filepath = STATIC_IMAGES_DIR / filename
                if not filepath.exists():
                    # 创建一个渐变色图片
                    img = Image.new('RGB', (1024, 768))
                    draw = ImageDraw.Draw(img)
                    
                    # 绘制渐变背景
                    for y in range(768):
                        # 从绿色渐变到蓝色（模拟森林天空）
                        r = int(50 + (y / 768) * 100)
                        g = int(150 - (y / 768) * 50)
                        b = int(100 + (y / 768) * 155)
                        draw.rectangle([(0, y), (1024, y+1)], fill=(r, g, b))
                    
                    # 添加文字
                    try:
                        draw.text((512, 384), f"Forest Scene {i+1}", 
                                fill=(255, 255, 255), anchor="mm")
                    except:
                        pass  # 如果没有字体也没关系
                    
                    img.save(filepath, 'JPEG', quality=95)
                    print(f"创建默认图片: {filepath}")
        except ImportError:
            print("PIL未安装，使用备用方案")
            # 如果没有PIL，创建一个占位文件
            for filename in self.default_images:
                filepath = STATIC_IMAGES_DIR / filename
                if not filepath.exists():
                    # 创建一个空文件作为占位符
                    filepath.write_text("placeholder")
    
    def get_next_image(self):
        """
        获取下一张图片的URL。
        修改后：不再复制文件，总是返回一个固定的静态图片。
        """
        # if not self.static_images:
        #     return None
        
        # # 获取下一张图片的路径
        # image_path = self.static_images[self.current_index]
        # self.current_index = (self.current_index + 1) % len(self.static_images)
        
        # # 生成新的文件名并定义目标路径
        # new_filename = f"generated_image_{uuid.uuid4().hex[:8]}.jpg"
        # destination_path = OUTPUTS_DIR / new_filename
        
        # # 复制图片到outputs目录
        # shutil.copy(image_path, destination_path)
        
        # # 返回新图片的URL
        # return f"/outputs/{new_filename}"

        # 直接返回一个固定的静态图片URL，不再执行任何文件写入操作
        return "/static_images/forest_scene_1.jpg"

@app.on_event("startup")
def startup_event():
    """获取下一张图片"""
    # 如果有真实的静态图片，使用它们
    # existing_images = list(STATIC_IMAGES_DIR.glob("*.jpg")) + \
    #                  list(STATIC_IMAGES_DIR.glob("*.png")) + \
    #                  list(STATIC_IMAGES_DIR.glob("*.jpeg"))
    
    # if existing_images:
    #     # 循环使用现有图片
    #     image_file = existing_images[self.counter % len(existing_images)]
    #     self.counter += 1
        
    #     # 复制到outputs目录
    #     image_id = str(uuid.uuid4())[:8]
    #     output_filename = f"generated_image_{image_id}{image_file.suffix}"
    #     output_path = OUTPUTS_DIR / output_filename
        
    #     try:
    #         shutil.copy2(image_file, output_path)
    #         return f"/outputs/{output_filename}"
    #     except Exception as e:
    #         print(f"复制图片失败: {e}")
    
    # 如果没有图片或复制失败，返回默认路径
    return "/outputs/sample_image.png"

# 初始化图片管理器
image_manager = ImageManager()

# ==================== 核心功能 ====================

async def generate_image_static(prompt: str, gpt_prompt: str, width: int = 1024, height: int = 768):
    """
    模拟生成图片（使用静态资源）
    
    Args:
        prompt: 场景描述
        gpt_prompt: 核心概念
        width: 图片宽度
        height: 图片高度
    
    Returns:
        dict: 包含图片URL的字典
    """
    print("=" * 60)
    print(f"[MCP1 Static] 开始生成图片")
    print(f"[MCP1 Static] 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MCP1 Static] 场景描述: {prompt[:100]}...")
    print(f"[MCP1 Static] GPT核心概念: {gpt_prompt[:100]}...")
    print(f"[MCP1 Static] 目标尺寸: {width}x{height}")
    print("=" * 60)
    
    # 模拟处理延迟（2-3秒）
    print(f"[MCP1 Static] 正在处理图片生成...")
    await asyncio.sleep(2.5)  # 延迟2.5秒
    
    # 获取下一张静态图片
    image_url = image_manager.get_next_image()
    
    print(f"[MCP1 Static] 生成成功!")
    print(f"[MCP1 Static] 图片URL: {image_url}")
    
    return {
        "local_path": image_url,
        "web_url": image_url
    }

# ==================== API端点 ====================

@app.post("/mcp1/text-to-image", response_model=TextToImageResponse)
async def text_to_image(request: TextToImageRequest):
    """
    MCP1标准接口：文本生成图片（静态实现）
    
    接收文本描述和GPT prompt，返回预设的图片
    """
    print("\n" + "=" * 80)
    print(f"[MCP1 Static API] 收到文生图请求")
    print(f"[MCP1 Static API] 请求时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MCP1 Static API] 请求参数:")
    print(f"  - prompt: {request.prompt[:100]}...")
    print(f"  - gpt_prompt: {request.gpt_prompt[:100]}...")
    print(f"  - width: {request.width}")
    print(f"  - height: {request.height}")
    print("=" * 80)
    
    try:
        # 调用静态生成函数
        result = await generate_image_static(
            prompt=request.prompt,
            gpt_prompt=request.gpt_prompt,
            width=request.width,
            height=request.height
        )
        
        # 构建响应
        response = TextToImageResponse(
            status="success",
            image_url=result["web_url"]
        )
        
        print(f"[MCP1 Static API] 响应成功")
        print(f"[MCP1 Static API] 返回图片URL: {response.image_url}")
        print("=" * 80 + "\n")
        
        return response
        
    except Exception as e:
        error_msg = f"生成失败: {str(e)}"
        print(f"[MCP1 Static API] 错误: {error_msg}")
        print("=" * 80 + "\n")
        
        # 即使出错也返回一个默认图片
        return TextToImageResponse(
            status="success",
            image_url="/outputs/sample_image.png"
        )

@app.get("/")
async def root():
    """服务信息"""
    return {
        "service": "MCP1 - Text to Image (Static)",
        "description": "文本生成图片服务（静态资源版）",
        "status": "running",
        "mode": "static",
        "endpoints": {
            "text_to_image": "/mcp1/text-to-image",
            "outputs": "/outputs/",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """健康检查"""
    # 检查静态图片
    static_images = []
    if STATIC_IMAGES_DIR.exists():
        for img in STATIC_IMAGES_DIR.glob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                static_images.append(img.name)
    
    return {
        "status": "healthy",
        "mode": "static",
        "outputs_dir": str(OUTPUTS_DIR),
        "temp_dir": str(TEMP_DIR),
        "static_images_dir": str(STATIC_IMAGES_DIR),
        "available_images": static_images,
        "image_count": len(static_images),
        "counter": image_manager.counter
    }

# ==================== 测试端点 ====================

@app.post("/test")
async def test_generation():
    """测试生成功能"""
    test_request = TextToImageRequest(
        prompt="宁静的森林景观，郁郁葱葱的绿色植物，参天大树和茂密的树叶",
        gpt_prompt="A cinematic forest scene with natural lighting",
        width=1024,
        height=768
    )
    
    return await text_to_image(test_request)

@app.get("/test-image")
async def test_image_display():
    """测试图片显示页面"""
    # 获取最新生成的图片
    output_images = list(OUTPUTS_DIR.glob("generated_image_*.png")) + \
                   list(OUTPUTS_DIR.glob("generated_image_*.jpg"))
    
    latest_image = None
    if output_images:
        latest_image = f"/outputs/{sorted(output_images)[-1].name}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP1 图片测试</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
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
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                margin: 20px 0;
            }}
            .info {{
                background: #e8f5e9;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .no-image {{
                padding: 40px;
                text-align: center;
                background: #fff3e0;
                border-radius: 8px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MCP1 图片测试（静态版）</h1>
            
            <div class="info">
                <p><strong>服务模式:</strong> 静态资源</p>
                <p><strong>处理延迟:</strong> 2.5秒（模拟）</p>
                <p><strong>图片尺寸:</strong> 1024×768</p>
            </div>
            
            {f'<img src="{latest_image}" alt="Generated Image">' if latest_image else '<div class="no-image">暂无生成的图片</div>'}
            
            <p style="text-align: center; color: #888; margin-top: 30px;">
                这是用于演示的静态图片资源
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
    print("MCP1 - 文本生成图片服务（静态资源版）")
    print("=" * 80)
    print(f"模式: 静态资源")
    print(f"输出目录: {OUTPUTS_DIR}")
    print(f"静态图片目录: {STATIC_IMAGES_DIR}")
    
    # 检查静态图片
    if STATIC_IMAGES_DIR.exists():
        images = list(STATIC_IMAGES_DIR.glob("*.jpg")) + \
                list(STATIC_IMAGES_DIR.glob("*.png")) + \
                list(STATIC_IMAGES_DIR.glob("*.jpeg"))
        print(f"可用静态图片: {len(images)}个")
        for img in images[:5]:
            print(f"  - {img.name}")
    else:
        print("静态图片目录不存在，将创建默认图片")
    
    print("\n端点:")
    print("  POST /mcp1/text-to-image - 文本生成图片")
    print("  GET  /outputs/           - 访问生成的图片")
    print("  POST /test               - 测试生成")
    print("  GET  /test-image         - 图片显示测试")
    print("  GET  /health             - 健康检查")
    print("=" * 80 + "\n")
    
    # 启动服务，监听8001端口，关闭自动重载
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)