from ingest import load_faq_data, build_index

documents = load_faq_data()
index = build_index(documents)

INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate answers.
If the context contains related or similar information, use it to infer
a reasonable answer — even if the wording doesn't match exactly.
Only respond with "I don't know" if the context has absolutely no relevant
information. Do not make up facts not supported by the context.
'''

USER_PROMPT_TEMPLATE ='''
Question: {question}

Context:{context}
'''

class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='llama-3.3-70b-versatile'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model
 

    def search(self, question):
        boost_dict = {'question': 2.0, 'section': 0.5}
        filter_dict = {'course': self.course}
        return self.index.search(question,
                                    boost_dict=boost_dict,
                                    filter_dict=filter_dict)

    def build_context(self, search_results) :
        lines = []

        for doc in search_results:
            lines.append(doc['section'])
            lines.append('Q: '+ doc['question' ])
            lines.append('A: '+ doc['answer'])
            lines.append('')

        return '\n'.join(lines). strip()

    def build_prompt(self, question, search_results):
        context = self.build_context(search_results)
        prompt = self.prompt_template.format(question=question, context=context)
        return prompt.strip()

    def llm(self, user_prompt):
        message_history = [
            {'role': 'system', 'content': self.instructions},
            {'role': 'user', 'content': user_prompt}
        ]
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=message_history
        )
        return response.choices[0].message.content

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer