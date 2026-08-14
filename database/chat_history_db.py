import sqlite3
import json
from datetime import datetime

from utils.logger import (
    rag_logger,
    error_logger,
    performance_logger
)

# ============================================================
# Configuration
# ============================================================

DB_PATH = "data/bridgebot.db"


# ============================================================
# Database Connection
# ============================================================

def get_connection():
    """
    Create SQLite database connection.
    """

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# Initialize Chat Tables
# ============================================================

def init_chat_history_tables():

    try:

        conn = get_connection()

        cursor = conn.cursor()

        # ----------------------------------------------------
        # Conversations
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id INTEGER NOT NULL,

                title TEXT NOT NULL,

                status TEXT DEFAULT 'ACTIVE',

                created_at TEXT NOT NULL,

                updated_at TEXT NOT NULL,

                FOREIGN KEY (user_id)
                    REFERENCES users(id)
            )
            """
        )

        # ----------------------------------------------------
        # Messages
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                conversation_id INTEGER NOT NULL,

                user_id INTEGER NOT NULL,

                role TEXT NOT NULL,

                content TEXT NOT NULL,

                source TEXT,

                metadata TEXT,

                created_at TEXT NOT NULL,

                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id),

                FOREIGN KEY (user_id)
                    REFERENCES users(id)
            )
            """
        )

        # ----------------------------------------------------
        # Indexes
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_conversations_user_id
            ON conversations(user_id)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_messages_conversation_id
            ON messages(conversation_id)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_messages_user_id
            ON messages(user_id)
            """
        )

        conn.commit()

        conn.close()

        rag_logger.info(
            "Chat history tables initialized successfully"
        )

    except Exception as ex:

        error_logger.exception(
            "Chat history table initialization failed : %s",
            ex
        )

        raise


# ============================================================
# Create Conversation
# ============================================================

def create_conversation(user_id, title="New Chat"):

    try:

        now = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO conversations
            (
                user_id,
                title,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                title,
                "ACTIVE",
                now,
                now
            )
        )

        conversation_id = cursor.lastrowid

        conn.commit()

        conn.close()

        rag_logger.info(
            "Conversation Created | user_id=%s | conversation_id=%s",
            user_id,
            conversation_id
        )

        return conversation_id

    except Exception as ex:

        error_logger.exception(
            "Conversation creation failed : %s",
            ex
        )

        return None


# ============================================================
# Get User Conversations
# ============================================================

def get_user_conversations(user_id):

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                user_id,
                title,
                status,
                created_at,
                updated_at
            FROM conversations
            WHERE user_id = ?
            AND status = 'ACTIVE'
            ORDER BY updated_at DESC
            """,
            (user_id,)
        )

        rows = cursor.fetchall()

        conn.close()

        return [dict(row) for row in rows]

    except Exception as ex:

        error_logger.exception(
            "Failed to load conversations : %s",
            ex
        )

        return []


# ============================================================
# Get Conversation
# ============================================================

def get_conversation(
    conversation_id,
    user_id
):

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                user_id,
                title,
                status,
                created_at,
                updated_at
            FROM conversations
            WHERE id = ?
            AND user_id = ?
            """,
            (
                conversation_id,
                user_id
            )
        )

        row = cursor.fetchone()

        conn.close()

        if row:

            return dict(row)

        return None

    except Exception as ex:

        error_logger.exception(
            "Failed to load conversation : %s",
            ex
        )

        return None


# ============================================================
# Update Conversation Timestamp
# ============================================================

def update_conversation_timestamp(
    conversation_id,
    user_id
):

    try:

        now = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            AND user_id = ?
            """,
            (
                now,
                conversation_id,
                user_id
            )
        )

        conn.commit()

        conn.close()

    except Exception as ex:

        error_logger.exception(
            "Failed to update conversation timestamp : %s",
            ex
        )


# ============================================================
# Update Conversation Title
# ============================================================

def update_conversation_title(
    conversation_id,
    user_id,
    title
):

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE conversations
            SET title = ?,
                updated_at = ?
            WHERE id = ?
            AND user_id = ?
            """,
            (
                title,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                conversation_id,
                user_id
            )
        )

        conn.commit()

        conn.close()

    except Exception as ex:

        error_logger.exception(
            "Failed to update conversation title : %s",
            ex
        )


# ============================================================
# Save Message
# ============================================================

def save_message(
    conversation_id,
    user_id,
    role,
    content,
    source=None,
    metadata=None
):

    try:

        now = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        metadata_json = None

        if metadata is not None:

            metadata_json = json.dumps(
                metadata,
                default=str
            )

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO messages
            (
                conversation_id,
                user_id,
                role,
                content,
                source,
                metadata,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                user_id,
                role,
                content,
                source,
                metadata_json,
                now
            )
        )

        message_id = cursor.lastrowid

        # ----------------------------------------------------
        # Update conversation activity
        # ----------------------------------------------------

        cursor.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            AND user_id = ?
            """,
            (
                now,
                conversation_id,
                user_id
            )
        )

        conn.commit()

        conn.close()

        rag_logger.info(
            "Message Saved | conversation_id=%s | user_id=%s | role=%s",
            conversation_id,
            user_id,
            role
        )

        return message_id

    except Exception as ex:

        error_logger.exception(
            "Message save failed : %s",
            ex
        )

        return None


# ============================================================
# Get Conversation Messages
# ============================================================

def get_conversation_messages(
    conversation_id,
    user_id
):

    try:

        conn = get_connection()

        cursor = conn.cursor()

        # ----------------------------------------------------
        # Authorization Check
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT id
            FROM conversations
            WHERE id = ?
            AND user_id = ?
            """,
            (
                conversation_id,
                user_id
            )
        )

        conversation = cursor.fetchone()

        if not conversation:

            rag_logger.warning(
                "Unauthorized conversation access | "
                "user_id=%s | conversation_id=%s",
                user_id,
                conversation_id
            )

            conn.close()

            return []

        # ----------------------------------------------------
        # Load Messages
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                id,
                conversation_id,
                user_id,
                role,
                content,
                source,
                metadata,
                created_at
            FROM messages
            WHERE conversation_id = ?
            AND user_id = ?
            ORDER BY id ASC
            """,
            (
                conversation_id,
                user_id
            )
        )

        rows = cursor.fetchall()

        conn.close()

        messages = []

        for row in rows:

            message = dict(row)

            if message["metadata"]:

                try:

                    message["metadata"] = json.loads(
                        message["metadata"]
                    )

                except Exception:

                    message["metadata"] = {}

            else:

                message["metadata"] = {}

            messages.append(message)

        return messages

    except Exception as ex:

        error_logger.exception(
            "Failed to load conversation messages : %s",
            ex
        )

        return []