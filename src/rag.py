import os
import logging
import uuid

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
import dotenv
from langchain_chroma import Chroma

dotenv.load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
retriever = vectorstore.as_retriever(k=4, fetch_k=20)

# Define the system prompt
SYSTEM_PROMPT = """
You are helpful ai assistant. Answer the user message. 

Knowledge Base:
{context}
""".strip()

model = ChatOpenAI(model="gpt-4o", temperature=0.2)


async def add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    url = update.message.text.split(" ", 1)[1]
    logging.info(f'add url {url}')
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    logging.info(f"Adding {len(splits)} documents to the knowledge base.")
    vectorstore.add_documents(splits, ids=[str(uuid.uuid4()) for _ in range(len(splits))])
    await update.message.reply_text(f"Added {len(splits)} documents to the knowledge base.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_question = update.message.text
    logging.info(f'USER:{update.effective_user.id} > {user_question}')
    docs = retriever.invoke(user_question)
    docs_text = "\n----\n".join(d.page_content for d in docs)
    logging.info(f'docs_text:\n{docs_text}')
    system_prompt_fmt = SYSTEM_PROMPT.format(context=docs_text)
    response = model.invoke([
        SystemMessage(content=system_prompt_fmt),
        HumanMessage(content=user_question)
    ])
    logging.info(f'BOT:{update.effective_user.id} > {response.content}')
    await update.message.reply_text(response.content)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error(f"Exception while handling an update: {context.error}")
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(f"Error: {context.error}")


def main():
    application = ApplicationBuilder().token(os.environ['TELEGRAM_BOT_TOKEN']).build()
    application.add_handler(CommandHandler("add", add))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    application.run_polling()


if __name__ == "__main__":
    main()
