from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import logging
import asyncio
from functools import partial
import concurrent.futures

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载模型
def load_model():
    logger.info("正在加载模型...")
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 设置特殊标记
        tokenizer.pad_token = tokenizer.eos_token
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )
        logger.info("模型加载成功")
        return pipe
    except Exception as e:
        logger.error(f"模型加载失败：{e}")
        raise e

pipe = load_model()

# 构建chat template
def build_prompt(user_message):
    return f"""<|system|>
You are a highly knowledgeable AI assistant. When asked about a topic, you provide detailed, comprehensive explanations of at least 150 words. Your responses include scientific facts, interesting details, and real-world examples. You maintain a clear and organized structure in your explanations.

<|user|>
{user_message}

<|assistant|>"""

# 在单独的线程中运行模型推理
def run_inference(prompt):
    logger.info("开始模型推理...")
    try:
        outputs = pipe(
            prompt, 
            max_new_tokens=512,  # 增加生成的token数量
            do_sample=True, 
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=pipe.tokenizer.eos_token_id,
            repetition_penalty=1.2,  # 增加重复惩罚
            num_beams=4,  # 使用beam search
            length_penalty=1.5,  # 鼓励生成更长的回答
            num_return_sequences=1
        )
        logger.info("模型推理完成")
        return outputs
    except Exception as e:
        logger.error(f"模型推理失败：{e}")
        raise e

# 清理模型输出
def clean_response(text):
    # 移除系统和用户提示部分
    if '<|assistant|>' in text:
        text = text.split('<|assistant|>')[-1]
    
    # 移除可能的后续对话标记
    if '<|user|>' in text:
        text = text.split('<|user|>')[0]
    
    # 移除可能的结束标记
    if '<|endoftext|>' in text:
        text = text.split('<|endoftext|>')[0]
        
    # 整理格式
    text = text.strip()
    # 如果回答少于3个词，认为是无效回答
    if len(text.split()) < 3:
        return ""
        
    return text

# 启动命令
async def start(update: Update, context):
    await update.message.reply_text(
        "Hello! I'm your AI assistant powered by TinyLlama. I can provide detailed information about various topics. What would you like to know about?"
    )

# 处理用户消息并调用 LLM
async def handle_message(update: Update, context):
    user_message = update.message.text
    logger.info(f"收到消息：{user_message}")

    try:
        # 发送等待消息
        waiting_message = await update.message.reply_text("让我来详细回答这个问题...")
        
        # 构建prompt
        logger.info("正在生成prompt...")
        prompt = build_prompt(user_message)
        logger.info(f"生成的prompt：{prompt}")
        
        # 在线程池中运行模型推理
        logger.info("正在调用模型...")
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            try:
                outputs = await asyncio.wait_for(
                    loop.run_in_executor(pool, partial(run_inference, prompt)),
                    timeout=500
                )
                logger.info(f"模型推理成功，原始输出：{outputs}")
            except asyncio.TimeoutError:
                logger.error("模型推理超时")
                await waiting_message.delete()
                await update.message.reply_text("抱歉，思考时间太长了，请稍后再试。")
                return
            except Exception as e:
                logger.error(f"模型推理出错：{e}")
                await waiting_message.delete()
                await update.message.reply_text("抱歉，处理您的请求时出现了问题，请稍后再试。")
                return

        # 提取和清理模型的回复
        try:
            response = outputs[0]["generated_text"]
            clean_text = clean_response(response)
            logger.info(f"清理后的回复：{clean_text}")
            
            if clean_text:
                await waiting_message.delete()
                await update.message.reply_text(clean_text)
            else:
                logger.warning("生成的回复为空或无效")
                await waiting_message.delete()
                await update.message.reply_text("抱歉，我没有生成有效的回复，请换个方式提问。")
        except Exception as e:
            logger.error(f"处理回复时出错：{e}")
            await waiting_message.delete()
            await update.message.reply_text("抱歉，处理回复时出现了问题。")
            
    except Exception as e:
        logger.error(f"处理消息时出错：{e}")
        await update.message.reply_text("抱歉，处理您的请求时出现了问题，请稍后再试。")

# 错误处理函数
async def error_handler(update, context):
    logger.error(f"更新 {update} 导致错误 {context.error}")

# 主函数
def main():
    # 测试模型
    try:
        logger.info("正在测试模型...")
        test_prompt = build_prompt("What is the capital of France?")
        test_output = pipe(test_prompt, max_new_tokens=50)
        logger.info(f"模型测试成功：{test_output}")
    except Exception as e:
        logger.error(f"模型测试失败：{e}")
        return

    API_TOKEN = "8068745258:AAFHdCgyw7jyhiN95q2Sry9Dm_0qgmMn3T8"
    
    # 创建应用
    try:
        app = Application.builder().token(API_TOKEN).build()
    except Exception as e:
        logger.error(f"创建应用失败：{e}")
        return

    # 添加处理器
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    # 启动机器人
    logger.info("机器人已启动，按 Ctrl+C 停止")
    try:
        app.run_polling()
    except Exception as e:
        logger.error(f"运行时错误：{e}")

if __name__ == "__main__":
    main()
