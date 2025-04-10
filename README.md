# ThirukuralGPT
I have created ThirukuralGPT for [Thirukural](https://en.wikipedia.org/wiki/Kural) inspired by Andrej Karpathy's video

I have downloaded the Thirukural from here, and cleaned little bit and used as the Dataset and used Kaggle for Training in GPU 

As we say GPT is Decoder only Transformer so, I have written Written Vectorization like Token Embedding, Positional Encoding and Decoder Block, Generation Function in Python Program using PyTorch and started Training

Initially our Model was generating "அறமல்லுலகத்துண்ணாதா லவற்கு" like those

And we need to do some Hyperparameter Tuning to get the Thirukural like words and sentences

So, after doing some Hyperparameter Tuning, Rugularization, changing Block Size and Training even more, I started to see the words the GPT is generating like below,

நாணாசென் போர்த்து கண்தவின் வஞுத்தீன்து தக்றத் தாகும் அண்.

நாணதாளில் கறிப்பதாத ல்லவார்க்கம் என்னிக் சிற்றென்றுப்லாரின் என்றாப் படனை.

ஈழியுப் பிறினுண டெல்லக்கி அடந்தாக்கல் பொற்றோர் பொய்வாக் குணை.

கூன்றுடு களப்படும் என்றால் நோக்கானும் அஃத முதனை.

இனந்துமை மழைந்தூர்க் கொள்ளவன் நாணோக்கு உலகத்து.

தனைந்துஞ்சால் கல்லவி நல்லயம் பயன்னும் செல்தொறித்தி யார தலை.

பட்டந்தபும் என்னார்நாடு நீர்.

 

Note, at this point in training our Model started to know, that Thirukural only have Seven Words in a Sentence and our Model also tried to reproduce the same(yes, in some sentences it has few or more than seven words, but it figures beautifully, right) 
