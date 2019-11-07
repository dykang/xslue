# xSLUE
Data and code for ["xSLUE: A Benchmark and Analysis Platform for \\Cross-Style Language Understanding and Evaluation
"](https://arxiv.org) by Dongyeop Kang and Eduard Hovy. If you have any questions, please contact to Dongyeop Kang (dongyeok@cs.cmu.edu).

We provide an online platform ([xSLUE.com](http://xslue.com/)) for cross-style language understanding and evaluation.
The benchmark contains 15 different styles and 23 classification tasks. For each task, we also provide the fine-tuned BERT classifier for further analysis. Our analysis shows that some styles are highly dependent on each other (e.g., impoliteness and offense), and some domains (e.g., tweets, political debates) are stylistically more diverse than the others (e.g., academic manuscripts).


## Citation

    @inproceedings{kang19arxiv_xslue,
        title = {xSLUE: A Benchmark and Analysis Platform for \\Cross-Style Language Understanding and Evaluation},
        author = {Dongyeop Kang and Eduard Hovy},
        booktitle = {https://arxiv.org},
        url = {https://arxiv.org},
        year = {2019}
    }

### Note
- The diagnostic set is only available upon request, since thie work is under review. We will publicly release it upon acceptance.
 - 

### Installation
Please download the pre-processed dataset and fine-tuned BERT classifier for each style in the [task](http://xslue.com/) tab. 


Every corpora has the same format of dataset as follow:
```
Dataset format: 
[source sentences] \t [target sentences]
or
<s> I was at home .. </s> <s> It was rainy day ..</s> ... \t <s> Sleeping at home rainy day </s> ..
```
An example python script for loading each dataset is provided here
```
python example/data_load.py --dataset AMI
```
### xSLUE Benmark
(please check [task] tab for more details in [BiasSum.com](http://biassum.com))
 - Formality GYAFC Not public [original](https://github.com/raosudha89/GYAFC-corpus) [classifier](https://github.com/dykang/xslue)


### Leaderboard
 - Please contact to Dongyeop if you like to add your cross-style system to the leaderboard. 
