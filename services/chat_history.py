from database.chat_history_db import (
    init_chat_history_tables,
    create_conversation,
    get_user_conversations,
    get_conversation,
    get_conversation_messages,
    save_message,
    update_conversation_title
)

from utils.logger import (
    rag_logger,
    error_logger
)


# ============================================================
# Initialize
# ============================================================

def initialize_chat_history():

    try:

        init_chat_history_tables()

        rag_logger.info(
            "Chat History Service Initialized"
        )

    except Exception as ex:

        error_logger.exception(
            "Chat History Service Initialization Failed : %s",
            ex
        )

        raise


# ============================================================
# Create New Chat
# ============================================================

def new_chat(
    user_id,
    title="New Chat"
):

    return create_conversation(
        user_id=user_id,
        title=title
    )


# ============================================================
# Load User Chats
# ============================================================

def load_user_chats(user_id):

    return get_user_conversations(
        user_id
    )


# ============================================================
# Load Chat
# ============================================================

def load_chat(
    conversation_id,
    user_id
):

    return get_conversation_messages(
        conversation_id,
        user_id
    )


# ============================================================
# Save User Message
# ============================================================

def save_user_message(
    conversation_id,
    user_id,
    content
):

    return save_message(
        conversation_id=conversation_id,
        user_id=user_id,
        role="user",
        content=content
    )


# ============================================================
# Save Assistant Message
# ============================================================

def save_assistant_message(
    conversation_id,
    user_id,
    content,
    source=None,
    metadata=None
):

    return save_message(
        conversation_id=conversation_id,
        user_id=user_id,
        role="assistant",
        content=content,
        source=source,
        metadata=metadata
    )


# ============================================================
# Update Title
# ============================================================

def rename_chat(
    conversation_id,
    user_id,
    title
):

    update_conversation_title(
        conversation_id,
        user_id,
        title
    )