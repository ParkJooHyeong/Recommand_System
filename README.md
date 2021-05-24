# Recommand_System
학교 수업에서 배운 수치해석을 활용, Kaggle의 데이터를 기반으로 영화 추천 시스템을 만들어 본다.   
SVD(Singular Value Decomposition), Machine Learning, Deep Learning 등 다양한 방법으로 추천 시스템을 구현해 본다.

- [컨텐츠 기반 협업 필터링(Item based Collaborative Filtering)](#Item-based-Collaborative-Filtering)
- [잠재요인 협업 필터링(Latent factor Collaborative Filtering)](#Latent-factor-Collaborative-Filtering)
- [잠재요인 협업 필터링 part2 (Latent factor Collaborative Filtering)](#Latent-factor-Collaborative-Filtering-part2)

------------
## Item based Collaborative Filtering
Dataset에서 rating(평점)을 활용해 입력된 영화와 평점이 유사한 영화를 뽑아서 추천해 본다.   
즉 Item based에서 Item을 rating으로 가정하고 접근한다.    
유사한 데이터를 수치적으로 얻기 위하여 코사인 유사도(cosine similarity)를 활용한다.     

~~~ python
from sklearn.metrics.pairwise import cosine_similarity
collabor = cosine_similarity(mv_to_user)
~~~

<img width="600" alt="cosine_sim" src="https://user-images.githubusercontent.com/67997760/119283533-58645580-bc78-11eb-8a91-cecaf4753a2c.png">  
cosine similarity의 경우 1에 가까울 수록 유사하다고 할 수 있다.     
Output : 가디언즈 오브 갤럭시를 검색한 결과.     
<img width="300" alt="스크린샷 2021-05-24 오전 9 51 39" src="https://user-images.githubusercontent.com/67997760/119282705-a4fa6180-bc75-11eb-83b7-38c7806a1329.png">


## Latent factor Collaborative Filtering
SVD(Singular Value Decomposition)을 활용.  
다수의 feature로 인한 다차원 데이터에서 차원(Dimension)을 축소해 행렬의 잠재요인을 찾아낸다.    
A = UΣVT  
![240px-Singular_value_decomposition_visualisation svg](https://user-images.githubusercontent.com/67997760/119285427-f9eda600-bc7c-11eb-966b-217d05fed34b.png)

~~~ python
from sklearn.decomposition import TruncatedSVD
SVD = TruncatedSVD(n_component=12)
~~~


## Latent factor Collaborative Filtering part2

------------
### Reference
Dataset : https://www.kaggle.com/rounakbanik/the-movies-dataset   
Blog : https://lsjsj92.tistory.com/   
Solution : https://www.kaggle.com/rounakbanik/movie-recommender-systems
