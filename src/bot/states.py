from enum import IntEnum


class ConversationState(IntEnum):
    IDLE = 0
    AWAITING_QUERY = 1