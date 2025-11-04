import { Movie, CarouselSection } from '../types/movie';

export const mockMovies: Movie[] = [
  {
    id: '1',
    title: 'Interstellar',
    titleKo: '인터스텔라',
    description: '지구의 종말이 다가오고, 인류를 구하기 위해 우주로 떠나는 탐험대의 이야기',
    year: 2014,
    director: '크리스토퍼 놀란',
    actors: ['매튜 매커너히', '앤 해서웨이', '제시카 차스테인'],
    genres: ['SF', '드라마', '모험'],
    poster: 'https://images.unsplash.com/photo-1585575141647-c2c436949374?w=400',
    backdrop: 'https://images.unsplash.com/photo-1585575141647-c2c436949374?w=1200'
  },
  {
    id: '2',
    title: 'Parasite',
    titleKo: '기생충',
    description: '전원 백수인 기택 가족이 IT 재벌 박 사장 가족에게 접근하면서 벌어지는 이야기',
    year: 2019,
    director: '봉준호',
    actors: ['송강호', '이선균', '조여정', '최우식'],
    genres: ['드라마', '스릴러', '코미디'],
    poster: 'https://images.unsplash.com/photo-1698881065188-1cef8476f33e?w=400',
    backdrop: 'https://images.unsplash.com/photo-1698881065188-1cef8476f33e?w=1200'
  },
  {
    id: '3',
    title: 'The Shawshank Redemption',
    titleKo: '쇼생크 탈출',
    description: '무고하게 종신형을 선고받은 은행가가 감옥에서 희망을 잃지 않는 이야기',
    year: 1994,
    director: '프랭크 다라본트',
    actors: ['팀 로빈스', '모건 프리먼'],
    genres: ['드라마'],
    poster: 'https://images.unsplash.com/photo-1699492718844-d7bff49bd2d1?w=400',
    backdrop: 'https://images.unsplash.com/photo-1699492718844-d7bff49bd2d1?w=1200'
  },
  {
    id: '4',
    title: 'Inception',
    titleKo: '인셉션',
    description: '꿈 속에서 생각을 훔치는 특수 요원이 마지막 임무를 수행하는 이야기',
    year: 2010,
    director: '크리스토퍼 놀란',
    actors: ['레오나르도 디카프리오', '조셉 고든 레빗', '엘런 페이지'],
    genres: ['SF', '액션', '스릴러'],
    poster: 'https://images.unsplash.com/photo-1551089568-6d63a7ee8afc?w=400',
    backdrop: 'https://images.unsplash.com/photo-1551089568-6d63a7ee8afc?w=1200'
  },
  {
    id: '5',
    title: 'The Dark Knight',
    titleKo: '다크 나이트',
    description: '배트맨이 조커와 맞서 싸우며 고담시를 지키는 이야기',
    year: 2008,
    director: '크리스토퍼 놀란',
    actors: ['크리스챤 베일', '히스 레저', '게리 올드만'],
    genres: ['액션', '범죄', '드라마'],
    poster: 'https://images.unsplash.com/photo-1665867534990-1f2e5787a3ee?w=400',
    backdrop: 'https://images.unsplash.com/photo-1665867534990-1f2e5787a3ee?w=1200'
  },
  {
    id: '6',
    title: 'Pulp Fiction',
    titleKo: '펄프 픽션',
    description: '로스앤젤레스 범죄 세계의 여러 이야기가 얽히는 블랙 코미디',
    year: 1994,
    director: '쿠엔틴 타란티노',
    actors: ['존 트라볼타', '사무엘 L. 잭슨', '우마 서먼'],
    genres: ['범죄', '드라마'],
    poster: 'https://images.unsplash.com/photo-1760981360579-ccc1969ded9e?w=400',
    backdrop: 'https://images.unsplash.com/photo-1760981360579-ccc1969ded9e?w=1200'
  },
  {
    id: '7',
    title: 'Memories of Murder',
    titleKo: '살인의 추억',
    description: '1980년대 한국의 연쇄살인 사건을 수사하는 형사들의 이야기',
    year: 2003,
    director: '봉준호',
    actors: ['송강호', '김상경', '김뢰하'],
    genres: ['범죄', '드라마', '스릴러'],
    poster: 'https://images.unsplash.com/photo-1624176194977-b0996f930188?w=400',
    backdrop: 'https://images.unsplash.com/photo-1624176194977-b0996f930188?w=1200'
  },
  {
    id: '8',
    title: 'The Matrix',
    titleKo: '매트릭스',
    description: '현실이 시뮬레이션임을 깨닫고 인류를 구하려는 해커의 이야기',
    year: 1999,
    director: '워쇼스키 자매',
    actors: ['키아누 리브스', '로렌스 피시번', '캐리 앤 모스'],
    genres: ['SF', '액션'],
    poster: 'https://images.unsplash.com/photo-1706245405770-ed9151cad554?w=400',
    backdrop: 'https://images.unsplash.com/photo-1706245405770-ed9151cad554?w=1200'
  },
  {
    id: '9',
    title: 'Oldboy',
    titleKo: '올드보이',
    description: '15년간 감금당한 남자가 복수를 위해 진실을 찾아가는 이야기',
    year: 2003,
    director: '박찬욱',
    actors: ['최민식', '유지태', '강혜정'],
    genres: ['액션', '드라마', '미스터리'],
    poster: 'https://images.unsplash.com/photo-1708805931229-35345161c2a5?w=400',
    backdrop: 'https://images.unsplash.com/photo-1708805931229-35345161c2a5?w=1200'
  },
  {
    id: '10',
    title: 'Arrival',
    titleKo: '컨택트',
    description: '외계인과 소통하기 위해 고용된 언어학자의 이야기',
    year: 2016,
    director: '드니 빌뇌브',
    actors: ['에이미 아담스', '제레미 레너', '포레스트 휘태커'],
    genres: ['SF', '드라마', '미스터리'],
    poster: 'https://images.unsplash.com/photo-1645526414400-9abff4d3ec0a?w=400',
    backdrop: 'https://images.unsplash.com/photo-1645526414400-9abff4d3ec0a?w=1200'
  },
  {
    id: '11',
    title: 'La La Land',
    titleKo: '라라랜드',
    description: '꿈을 쫓는 배우와 재즈 피아니스트의 사랑 이야기',
    year: 2016,
    director: '데이미언 차젤레',
    actors: ['라이언 고슬링', '엠마 스톤'],
    genres: ['로맨스', '뮤지컬', '드라마'],
    poster: 'https://images.unsplash.com/photo-1687589891979-d17de92e5802?w=400',
    backdrop: 'https://images.unsplash.com/photo-1687589891979-d17de92e5802?w=1200'
  },
  {
    id: '12',
    title: 'Whiplash',
    titleKo: '위플래쉬',
    description: '완벽을 추구하는 드럼 연주자와 가혹한 교수의 이야기',
    year: 2014,
    director: '데이미언 차젤레',
    actors: ['마일즈 텔러', 'J.K. 시몬스'],
    genres: ['드라마', '음악'],
    poster: 'https://images.unsplash.com/photo-1721480345171-8b34a1f34b95?w=400',
    backdrop: 'https://images.unsplash.com/photo-1721480345171-8b34a1f34b95?w=1200'
  },
  {
    id: '13',
    title: 'Forrest Gump',
    titleKo: '포레스트 검프',
    description: 'IQ 75의 순수한 남자가 미국 현대사의 중요한 순간들을 경험하는 이야기',
    year: 1994,
    director: '로버트 저메키스',
    actors: ['톰 행크스', '로빈 라이트', '게리 시나이즈'],
    genres: ['드라마', '로맨스'],
    poster: 'https://images.unsplash.com/photo-1631618786030-803b0f28a94e?w=400',
    backdrop: 'https://images.unsplash.com/photo-1631618786030-803b0f28a94e?w=1200'
  },
  {
    id: '14',
    title: 'The Godfather',
    titleKo: '대부',
    description: '마피아 가문의 권력 승계를 그린 범죄 드라마의 걸작',
    year: 1972,
    director: '프란시스 포드 코폴라',
    actors: ['말론 브란도', '알 파치노', '제임스 칸'],
    genres: ['범죄', '드라마'],
    poster: 'https://images.unsplash.com/photo-1674302605734-d5f430265a17?w=400',
    backdrop: 'https://images.unsplash.com/photo-1674302605734-d5f430265a17?w=1200'
  },
  {
    id: '15',
    title: 'Blade Runner 2049',
    titleKo: '블레이드 러너 2049',
    description: '레플리칸트를 쫓는 블레이드 러너가 오래된 비밀을 발견하는 이야기',
    year: 2017,
    director: '드니 빌뇌브',
    actors: ['라이언 고슬링', '해리슨 포드', '아나 드 아르마스'],
    genres: ['SF', '스릴러'],
    poster: 'https://images.unsplash.com/photo-1706245405770-ed9151cad554?w=400',
    backdrop: 'https://images.unsplash.com/photo-1706245405770-ed9151cad554?w=1200'
  },
  {
    id: '16',
    title: 'The Silence of the Lambs',
    titleKo: '양들의 침묵',
    description: 'FBI 요원이 연쇄살인범을 잡기 위해 수감된 식인 살인마와 대화하는 이야기',
    year: 1991,
    director: '조나단 뎀',
    actors: ['조디 포스터', '안소니 홉킨스', '스콧 글렌'],
    genres: ['스릴러', '범죄', '드라마'],
    poster: 'https://images.unsplash.com/photo-1578053612724-d4f2258ff76f?w=400',
    backdrop: 'https://images.unsplash.com/photo-1578053612724-d4f2258ff76f?w=1200'
  },
  {
    id: '17',
    title: 'Mad Max: Fury Road',
    titleKo: '매드 맥스: 분노의 도로',
    description: '황폐한 미래 세계에서 펼쳐지는 추격전과 생존의 이야기',
    year: 2015,
    director: '조지 밀러',
    actors: ['톰 하디', '샤를리즈 테론', '니콜라스 홀트'],
    genres: ['액션', 'SF', '모험'],
    poster: 'https://images.unsplash.com/photo-1738193026608-df7a79f36cd3?w=400',
    backdrop: 'https://images.unsplash.com/photo-1738193026608-df7a79f36cd3?w=1200'
  },
  {
    id: '18',
    title: 'Her',
    titleKo: '그녀',
    description: '인공지능 OS와 사랑에 빠진 남자의 이야기',
    year: 2013,
    director: '스파이크 존즈',
    actors: ['호아킨 피닉스', '스칼렛 요한슨', '에이미 아담스'],
    genres: ['로맨스', 'SF', '드라마'],
    poster: 'https://images.unsplash.com/photo-1650538230322-069d33d0fdbe?w=400',
    backdrop: 'https://images.unsplash.com/photo-1650538230322-069d33d0fdbe?w=1200'
  },
  {
    id: '19',
    title: 'The Host',
    titleKo: '괴물',
    description: '한강에 나타난 괴물에게 딸을 잃은 가족의 사투를 그린 이야기',
    year: 2006,
    director: '봉준호',
    actors: ['송강호', '변희봉', '박해일', '배두나'],
    genres: ['SF', '드라마', '액션'],
    poster: 'https://images.unsplash.com/photo-1761325684397-b91138faca5f?w=400',
    backdrop: 'https://images.unsplash.com/photo-1761325684397-b91138faca5f?w=1200'
  },
  {
    id: '20',
    title: 'Gone Girl',
    titleKo: '나를 찾아줘',
    description: '실종된 아내를 찾는 과정에서 드러나는 충격적인 진실',
    year: 2014,
    director: '데이비드 핀처',
    actors: ['벤 애플렉', '로자먼드 파이크', '닐 패트릭 해리스'],
    genres: ['스릴러', '미스터리', '드라마'],
    poster: 'https://images.unsplash.com/photo-1547637974-a0d8a38ebbda?w=400',
    backdrop: 'https://images.unsplash.com/photo-1547637974-a0d8a38ebbda?w=1200'
  }
];

// Create placeholder movies for carousels
const createPlaceholderMovies = (count: number, prefix: string): Movie[] => {
  return Array.from({ length: count }, (_, i) => ({
    id: `${prefix}-${i + 1}`,
    title: `Movie ${i + 1}`,
    titleKo: `영화 ${i + 1}`,
    description: 'Placeholder movie description',
    year: 2024,
    director: 'Director',
    actors: ['Actor 1', 'Actor 2'],
    genres: ['Genre'],
    poster: '',
    backdrop: ''
  }));
};

export const initialCarousels: CarouselSection[] = [
  {
    id: 'sf',
    title: 'SF',
    movies: createPlaceholderMovies(10, 'sf')
  },
  {
    id: 'popular',
    title: '다른 사람이 좋아하는',
    movies: createPlaceholderMovies(10, 'popular')
  },
  {
    id: 'bong',
    title: '봉준호를 좋아하신다면',
    movies: createPlaceholderMovies(10, 'bong')
  },
  {
    id: 'nolan',
    title: '크리스토퍼 놀란 감독 작품',
    movies: createPlaceholderMovies(10, 'nolan')
  },
  {
    id: 'thriller',
    title: '긴장감 넘치는 스릴러',
    movies: createPlaceholderMovies(10, 'thriller')
  },
  {
    id: 'romance',
    title: '감성 충만 로맨스',
    movies: createPlaceholderMovies(10, 'romance')
  }
];
