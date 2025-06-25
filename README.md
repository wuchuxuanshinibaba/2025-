# 2025-云计算大作业
期末大作业第一问.py是用文本和情感分析结合大模型去做谣言检测；
期末大作业第二问.py主要是做文本的主题分析；
双重不一致谣言检测网络的文件夹则是利用双重不一致检测网络完成谣言检测任务。
本大作业只聚焦于任务实现，只利用少量数据运行代码。

第一、第二问：
原始数据集是Shu, Kai et al. “FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media.” Big data 8 3 (2018): 171-188 .
格式转换.py用于将原始数据的.txt转成.csv,仅仅只是方便浏览数据。
所用的大模型是通过ollama本地部署的gemma3。在本地部署大模型后，可以直接运行期末大作业第一问.py（我已将部分原始数据整理成验证集.csv）
期末大作业第二问.py可直接运行。lda_visualization.html、topic_distribution.png、topic_wordclouds.png是其结果。

第三问：
基础要求：32G的运行内存，独显（配置相应的CUDA和torch版本）
要求下载bert-base-中文文件（https://huggingface.co/google-bert/bert-base-chinese/tree/main）和bert-base-uncased的文件（https://huggingface.co/google-bert/bert-base-uncased/tree/main），以及http://openke.thunlp.org/resources/embedding/freebase.zip（知识图谱）
需要自行准备data文件夹（包含图片和csv文件，可通过最下方的引用论文进行获取（图片可根据csv中正文给出的网址得到））
令bert-base-chinese、bert-base-uncased、data、Freebase、model、process同级保存在一个文件夹中，注意最好不要使用中文字符命名文件或文件夹。
将相关文件的路径正确设置为你的电脑路径即可正常运行。
本网络最优秀的地方在于运行小数据集也不会出现梯度爆炸问题。

如果无法正常运行或出现梯度爆炸问题，请查看requirement.txt的相关依赖版本

双重不一致谣言检测网络引用自下面的论文：
@inproceedings{sun-etal-2021-inconsistency-matters,
    title = "Inconsistency Matters: A Knowledge-guided Dual-inconsistency Network for Multi-modal Rumor Detection",
    author = "Sun, Mengzhu  and
      Zhang, Xi  and
      Ma, Jianqiang  and
      Liu, Yazheng",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.122",
    pages = "1412--1423",
    abstract = "Rumor spreaders are increasingly utilizing multimedia content to attract the attention and trust of news consumers. Though a set of rumor detection models have exploited the multi-modal data, they seldom consider the inconsistent relationships among images and texts. Moreover, they also fail to find a powerful way to spot the inconsistency information among the post contents and background knowledge. Motivated by the intuition that rumors are more likely to have inconsistency information in semantics, a novel Knowledge-guided Dual-inconsistency network is proposed to detect rumors with multimedia contents. It can capture the inconsistent semantics at the cross-modal level and the content-knowledge level in one unified framework. Extensive experiments on two public real-world datasets demonstrate that our proposal can outperform the state-of-the-art baselines.",
}
