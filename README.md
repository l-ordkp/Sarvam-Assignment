This assignment had 3 parts, so does my program has 3 endpoints.

**1st Endpoint (Process Query)**:- This is an endpoint whuch uses RAG. **This has 2 vector db's (one for text and other for images (diagrams))** thus retrieving appropriate text as well as diagrams for the query .
**This RAG is optimised with the help of DSPy module's ChainofThought** which helps in breaking complex prompts into simpler prompts.

**2nd Endpoint (AI-Agent)**:- This agent as stated in assignment can do 3 tasks:- 1) Call DB 2)**Get songs by artist (Used Spotify's api)** 3). Answer generic questions

**3rd Endoint (AI-Voice-Agent)**:- This agent does all the tasks as the same as 2nd Endpoint but returns the output **as the path where the response is stored in form of an audio file** (Target language and Speaker is hardcoded).
