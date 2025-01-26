from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# 定义处理 /start 命令的函数
async def start(update: Update, context):
    await update.message.reply_text("Hello！Yutong ~ I'm your assistant, send any message and I will reply！")

# 定义处理其他消息的函数
async def echo(update: Update, context):
    user_message = update.message.text
    await update.message.reply_text(f"你发送了：{user_message}")

# 主函数
def main():
    API_TOKEN = "8068745258:AAEwBkTb0qqO1yLWO9h10KBlCH2fUDosf6w"
    app = Application.builder().token(API_TOKEN).build()

    # 添加 /start 命令处理
    app.add_handler(CommandHandler("start", start))

    # 添加处理所有文本消息的处理器
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # 启动机器人
    print("机器人已启动，按 Ctrl+C 停止")
    app.run_polling()

if __name__ == "__main__":
    main()
