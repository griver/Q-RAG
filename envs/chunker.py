import re



def chunks_split(text:str, chunk_size=2000) -> list[str]:
    pattern = r"(?<=[.!?])\s+|(?<=\n\n)"
    sentences = [segment for segment in re.split(pattern, text) if segment]
    chunks = []
    current_chunk = []
    for sentence in sentences:
        if len(' '.join(current_chunk + [sentence])) <= chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks



if __name__ == "__main__":
    text = "This is sentence one. This is sentence two!\n\nNew paragraph here?"
    pat = r"(?<=[.!?])\s+|(?<=\n\n)"
    sentences = [segment for segment in re.split(pat, text) if segment]
    print(sentences)

    text = "000 sdfbsgs sdfdsg of advafvd ghrhryhhyrh 111. "*20
    chunks = chunks_split(text, chunk_size=200)
    print(chunks)
    print(len(chunks))
