from __future__ import annotations
from metagpt.actions import Action
from metagpt.schema import Message
PROMPT = """Given a conversation (between Human and Assistant) and a follow up message from Human, rewrite the message to be a standalone question that captures all relevant context from the conversation.

<Chat History>
   {chat_history}
   
<Follow Up Message>
   {query}
   
<Standalone question>
"""
class TransQuery(Action):
    async def run(self,
                    instruction: str ,
                    chat_history: str, 
                  ) -> str:
        
            prompt = PROMPT.format(chat_history = chat_history, query = instruction)
            context = self.llm.format_msg([Message(content=prompt, role="user")])
            rsp = await self.llm.aask(context)
            return rsp
        
