import numpy as np
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
import torch.utils
from nltk.probability import gt_demo
from torch.utils.data import Dataset

# from rl.jax_text_env import TextEnv, TextMemory, TextMemoryItem
from rl.text_env import TextEnv, TextMemory, TextMemoryItem
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    
class SimpleEnvAdapter(Dataset):
    """
    Simple adapter that adapts datasets Babilong, HotPotQA and MUSIQUE for QAREtreievalEnv.
    This adapter doesn't tokenize or embeds text chunks.

    You can create different adapter that for example tokenize every text in a sample or
    build faiss index over text chunks.
    """

    def __init__(self, dataset, min_chunks=6): # Добавили параметр min_chunks
        super().__init__()
        
        original_dataset = dataset
        self.dataset_name = original_dataset.name()
        
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        
        print(f"Фильтрация датасета '{self.dataset_name}'. Исходный размер: {len(original_dataset)}.")
        print(f"Удаляются сэмплы, где количество чанков (контекстных параграфов) меньше {min_chunks}.")
        
        filtered_dataset = []
        for sample in original_dataset:
            # Логика определения количества чанков должна соответствовать тому,
            # как они создаются в __getitem__. Для hotpotqa это len(sample['context']).
            num_chunks = 0
            if self.dataset_name == 'hotpotqa':
                num_chunks = len(sample.get('context', []))
            # elif self.dataset_name == 'musique':
            #     num_chunks = len(sample.get('paragraphs', []))
            # elif self.dataset_name == 'babilong':
            #     num_chunks = len(sample.get('chunks', []))
            
            if num_chunks >= min_chunks:
                filtered_dataset.append(sample)

        self.dataset = filtered_dataset
        
        print(f"Фильтрация завершена. Новый размер датасета: {len(self.dataset)}.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        question = sample["question"]
        if question.endswith("?"):
            question = question[:-1]

        sf_idx = []
        chunks_texts = []
        if self.dataset_name == 'hotpotqa':
            sp_title_set = set()
            sample_id = sample['_id']
            for sup in sample['supporting_facts']:
                sp_title_set.add(sup[0])

            for idx, (title, sentences) in enumerate(sample['context']):
                if title in sp_title_set:
                    sf_idx.append(idx)
                chunk = title + " " + " ".join(sentences)
                chunks_texts.append(chunk)

        elif self.dataset_name == 'musique':
            sample_id = sample['id']
            for i, para in enumerate(sample['paragraphs']):
                # if para['is_supporting']:
                #     sf_idx.append(i)
                chunk = para['title'] + '. ' + para['paragraph_text']
                chunks_texts.append(chunk)

            # label order
            for item_json in sample['question_decomposition']:
                sf_idx.append(item_json['paragraph_support_idx'])

        elif self.dataset_name == 'babilong':
            sample_id = index
            for i, sent in enumerate(sample['chunks']):
                chunks_texts.append(sent)

            for i in sample['references_idx']:
                sf_idx.append(i)

        return {
            'id': sample_id,
            'question': question,
            'answer': sample["answer"],
            'chunks_texts': chunks_texts,
            'sf_idx': sf_idx,
        }


class GroundTruthReward:
    def __init__(self, only_at_max_step=False):
        super().__init__()
        self.only_at_max_step = only_at_max_step

    def reward(self, env, action):
        if self.only_at_max_step and (env.num_steps < env.max_steps):
            return 0.

        is_retrieved = []
        for r in env.references:
            is_retrieved.append(r in env.text_state)

        all_retrieved = all(is_retrieved)
        return float(all_retrieved)


class PositionalGTReward(GroundTruthReward):
    """
    This version takes into account position of the support facts.
    In babi tasks several events could have completely identical text descriptions,
    but only one of them can be considered a support fact/reference fact.

    I.E. Merry could visit the same location several times.
    But only the last event allows us to tell where she is at the end of the story.

    This reward takes into account temporal information that allows to distinguish
    true support facts, from similar events.
    """
    def reward(self, env, action):
        if self.only_at_max_step and (env.num_steps < env.max_steps):
            return 0.

        pred_sf = set(map(int, env.memory.item_ids))
        gt_sf = set(env.references_idx)
        return 1.0 if gt_sf.issubset(pred_sf) else 0.0

class QARetrievalEnv(TextEnv):  # Наследуемся напрямую от TextEnv
    def __init__(self,
                 dataset: SimpleEnvAdapter, # Явно указываем тип для ясности
                 max_steps = 2,
                 index_type="random", # "absolute", "relative" - используется TextEnv?
                 reward_model=GroundTruthReward()):
        
        super().__init__() # Вызов конструктора TextEnv

        self.dataset = dataset # Это будет SimpleEnvAdapter
        self.max_steps = max_steps
        self.index_type = index_type # Если TextEnv его использует, иначе можно убрать
        self.reward_model = reward_model

        # Атрибуты, которые ранее устанавливались в BabilongEnv или его _init_from_sample
        # и используются в методах, которые мы скопируем или уже имеем
        self.references: List[str] = []
        self.question: str = ""
        self.answer: Any = None # Тип ответа может быть разным
        self.sentences: np.ndarray = np.array([]) # Массив текстов "чанков"
        self.facts_idx: List[int] = [] # Индексы "фактов" (в контексте Adapted это sf_idx)
        self.references_idx: List[int] = [] # Индексы релевантных чанков

        self.num_steps: int = 0
        self.text_state: List[str] = [] # Хранит тексты выбранных чанков

        # Атрибуты, которые могли быть в BabilongEnv и используются reward_model или др. логикой
        # self.refs_found = [] # Если используется где-то еще, кроме как локально в BabilongEnv.reset

    def _init_from_sample(self, sample: Dict[str, Any]):
        # 'sample' здесь приходит от SimpleEnvAdapter
        # Формат sample: {'id'(опционально), 'question', 'answer', 'chunks_texts', 'sf_idx'}
        
        self.question = sample['question']
        self.answer = sample['answer']
        self.sentences = np.asarray(sample['chunks_texts']) # Это будут наши "чанки"
        
        # 'sf_idx' из SimpleEnvAdapter - это индексы релевантных чанков в 'chunks_texts'
        # Это соответствует 'references_idx'
        self.references_idx = list(map(int, sample['sf_idx'])) # Убедимся, что это int
        
        # 'references' - это тексты релевантных чанков
        self.references = [self.sentences[i] for i in self.references_idx if 0 <= i < len(self.sentences)]
        
        # 'facts_idx' - для совместимости и если какая-то логика на них полагается.
        # В данном контексте "факты" это и есть поддерживающие чанки.
        self.facts_idx = list(self.references_idx) 
        
        # Опционально: отладочный вывод
        # print(f"QARetrievalEnv._init_from_sample: sample keys: {sample.keys()}")
        # print(f"  question: {self.question}")
        # print(f"  sentences len: {len(self.sentences)}")
        # print(f"  references_idx (from sf_idx): {self.references_idx}")
        # print(f"  facts_idx (set to sf_idx): {self.facts_idx}")

    def reset(self, new_sample: Dict[str, Any] = None) -> TextMemory:
        if new_sample is not None:
            self._init_from_sample(new_sample)
        elif self.dataset is not None:
            # Эта часть предполагает, что self.dataset (SimpleEnvAdapter)
            # поддерживает __len__ и __getitem__ для случайного выбора.
            # Если SimpleEnvAdapter предоставляет другой API (например, .sample()),
            # эту логику нужно будет адаптировать.
            try:
                N = len(self.dataset)
                if N == 0:
                    raise ValueError("Dataset is empty.")
                i = np.random.randint(N)
                sample_from_dataset = self.dataset[i]
                self._init_from_sample(sample_from_dataset)
            except (TypeError, NotImplementedError) as e:
                # Если __len__ или __getitem__ не реализованы, попробуем .sample()
                if hasattr(self.dataset, 'sample') and callable(getattr(self.dataset, 'sample')):
                    try:
                        sample_from_dataset = self.dataset.sample()
                        self._init_from_sample(sample_from_dataset)
                    except Exception as sample_e:
                        raise RuntimeError(f"Failed to get sample using .sample() from dataset: {sample_e}") from sample_e
                else:
                    raise RuntimeError(
                        "Dataset adapter does not support len()/getitem[] for sampling, "
                        "nor does it have a .sample() method. "
                        "Please provide new_sample directly to reset() or adapt the adapter."
                    ) from e
        else:
            # Если нет ни new_sample, ни dataset, это ошибка конфигурации
            raise ValueError("Cannot reset environment: no new_sample provided and no dataset configured.")


        self.num_steps = 0
        self.text_state = [] # Сбрасываем историю выбранных текстов
        self.refs_found = [] # Если этот атрибут был и использовался, его тоже сбрасываем

        # _reset из TextEnv должен инициализировать TextMemory на основе question и sentences
        # TextEnv._reset(self, query: str, chunks: Union[List[str], np.ndarray]) -> TextMemory
        return super()._reset(self.question, self.sentences)
   
    def step(self, action: int): # -> Tuple[TextMemory, TextMemoryItem, float, bool]
                                 # TextMemoryItem определен в rl.text_env
        self.num_steps += 1
        done = self.num_steps >= self.max_steps
        
        # _step из TextEnv должен обновить память и вернуть выбранный элемент
        # TextEnv._step(self, action_idx: int) -> Tuple[TextMemory, TextMemoryItem, bool]
        # где bool это text_done (например, если память переполнилась или все элементы выбраны)
        text_memory, text_item, text_done = super()._step(action)
        
        # Добавляем текст выбранного чанка в self.text_state для GroundTruthReward
        if 0 <= action < len(self.sentences):
            self.text_state.append(self.sentences[action])
        else:
            # Это не должно происходить, если action всегда валидный индекс
            # Можно добавить логгирование или обработку ошибки
            print(f"Warning: Action {action} is out of bounds for sentences (len: {len(self.sentences)})")


        r = self._reward(action)
        # Если достигнута цель (например, все референсы найдены), эпизод может завершиться досрочно
        if r > 1e-5: # Сравниваем с небольшим эпсилон, т.к. награда может быть float
            done = True
    
        return text_memory, text_item, r, done or text_done

    def _reward(self, action: int) -> float:
        # Используем self.reward_model, который был передан в конструкторе
        return self.reward_model.reward(self, action)

    @property
    def device(self):
        # Предполагаем, что TextEnv (родительский класс) управляет эмбеддером
        # и предоставляет доступ к его устройству.
        # Если TextEnv не имеет self.embedder, эту часть нужно адаптировать.
        if hasattr(self, 'embedder') and self.embedder is not None:
            return self.embedder.device
        # Заглушка или ошибка, если embedder не найден (зависит от реализации TextEnv)
        # print("Warning: embedder not found or not initialized in TextEnv for .device property.")
        return "cuda:0" # или None, или raise AttributeError

    def get_sample_len(self, tokenizer) -> int:
        """
        Return total length of all texts (question + chunks) in the current task,
        tokenized by the provided tokenizer.
        """
        if not self.question or self.sentences is None or len(self.sentences) == 0:
            # Это может случиться, если reset() не был вызван или не отработал корректно
            # print("Warning: get_sample_len called before environment was properly reset with data.")
            return 0
        
        total_len = len(tokenizer(self.question)['input_ids'])
        # Убедимся, что sentences - это список строк для токенизатора
        sentences_list = list(self.sentences)
        if sentences_list: # Проверка, что список не пустой
            # Некоторые токенизаторы могут возвращать список списков input_ids, если на вход список строк
            tokenized_chunks = tokenizer(sentences_list)['input_ids']
            total_len += sum(len(chunk_ids) for chunk_ids in tokenized_chunks)
        return total_len

    def copy(self):
        # Создаем копию этого же класса
        # self.dataset (SimpleEnvAdapter) и self.reward_model передаются по ссылке,
        # что обычно является ожидаемым поведением для таких объектов.
        # Если нужна глубокая копия этих объектов, ее нужно реализовать отдельно.
        return QARetrievalEnv(dataset=self.dataset,
                                  max_steps=self.max_steps,
                                  index_type=self.index_type,
                                  reward_model=self.reward_model)