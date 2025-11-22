'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { userAPI, actorsAPI, directorsAPI } from '@/lib/api';
import Header from '@/components/Header';

const GENRES = [
  '드라마', '액션', '코미디', '로맨스', '스릴러', '공포',
  'SF', '판타지', '애니메이션', '다큐멘터리', '범죄', '가족'
];

const genreMapReverse: Record<string, string> = {
  'Action': '액션', 'Drama': '드라마', 'Comedy': '코미디', 'Romance': '로맨스',
  'Thriller': '스릴러', 'Horror': '공포', 'Science Fiction': 'SF', 'Fantasy': '판타지',
  'Animation': '애니메이션', 'Documentary': '다큐멘터리', 'Crime': '범죄', 'Family': '가족'
};

export default function QuestionnairePage() {
  const router = useRouter();
  const { user, updateUser } = useAuth();
  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
  const [selectedActors, setSelectedActors] = useState<string[]>([]);
  const [selectedDirectors, setSelectedDirectors] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [loadingPrefs, setLoadingPrefs] = useState(true);

  useEffect(() => {
    if (user) {
      const koreanGenres = user.preferredGenres?.map(g => genreMapReverse[g] || g).filter(g => GENRES.includes(g)) || [];
      setSelectedGenres(koreanGenres);
      setSelectedActors(user.preferredActors || []);
      setSelectedDirectors(user.preferredDirectors || []);
    }
    setLoadingPrefs(false);
  }, [user]);

  const toggleGenre = (genre: string) => {
    setSelectedGenres(prev => prev.includes(genre) ? prev.filter(g => g !== genre) : [...prev, genre]);
  };

  const handleAddItem = (item: string, type: 'actor' | 'director') => {
    if (type === 'actor') {
      setSelectedActors(prev => [...new Set([...prev, item])]);
    } else {
      setSelectedDirectors(prev => [...new Set([...prev, item])]);
    }
  };

  const handleRemoveItem = (item: string, type: 'actor' | 'director') => {
    if (type === 'actor') {
      setSelectedActors(prev => prev.filter(i => i !== item));
    } else {
      setSelectedDirectors(prev => prev.filter(i => i !== item));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedGenres.length === 0) {
      setError('최소 1개의 장르를 선택해주세요');
      return;
    }
    setLoading(true);
    setError('');
    try {
      await updateUser({
        preferredGenres: selectedGenres,
        preferredActors: selectedActors,
        preferredDirectors: selectedDirectors,
      });
      router.push(sessionStorage.getItem('questionnaire_from_mypage') === 'true' ? '/user' : '/');
      sessionStorage.removeItem('questionnaire_from_mypage');
    } catch (err: any) {
      setError(err.response?.data?.error || '저장에 실패했습니다');
    } finally {
      setLoading(false);
    }
  };

  if (loadingPrefs) {
    return <div className="min-h-screen bg-black text-white"><Header /><div className="text-center py-16">설정을 불러오는 중...</div></div>;
  }

  return (
    <div className="min-h-screen bg-black text-white">
      <Header />
      <div className="container mx-auto px-4 pt-24 pb-16">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold mb-4">좋아하는 영화 취향을 알려주세요</h1>
            <p className="text-gray-400">더 정확한 추천을 위해 선호하는 장르, 배우, 감독을 알려주세요.</p>
          </div>
          <form onSubmit={handleSubmit}>
            <div className="bg-gray-900 rounded-lg p-8 mb-6">
              <h2 className="text-xl font-semibold mb-6">선호 장르 선택 (최소 1개)</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {GENRES.map(g => <button key={g} type="button" onClick={() => toggleGenre(g)} className={`px-6 py-3 rounded-lg border-2 transition ${selectedGenres.includes(g) ? 'bg-red-600 border-red-600' : 'bg-gray-800 border-gray-700 hover:border-red-500'}`}>{g}</button>)}
              </div>
            </div>
            <SearchableList title="선호 배우" items={selectedActors} onAdd={(item) => handleAddItem(item, 'actor')} onRemove={(item) => handleRemoveItem(item, 'actor')} searchFn={actorsAPI.search} />
            <SearchableList title="선호 감독" items={selectedDirectors} onAdd={(item) => handleAddItem(item, 'director')} onRemove={(item) => handleRemoveItem(item, 'director')} searchFn={directorsAPI.search} />
            {error && <div className="bg-red-500/10 border border-red-500 text-red-500 px-4 py-2 rounded my-6">{error}</div>}
            <div className="flex gap-4 mt-8">
              <button type="button" onClick={() => router.push('/')} className="flex-1 px-6 py-3 bg-gray-800 rounded-lg hover:bg-gray-700">나중에 하기</button>
              <button type="submit" disabled={loading || selectedGenres.length === 0} className="flex-1 px-6 py-3 bg-red-600 rounded-lg hover:bg-red-700 disabled:opacity-50">완료</button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

interface SearchableListProps {
  title: string;
  items: string[];
  onAdd: (item: string) => void;
  onRemove: (item: string) => void;
  searchFn: (query: string) => Promise<any>;
}

function SearchableList({ title, items, onAdd, onRemove, searchFn }: SearchableListProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);

  const handleSearch = async (q: string) => {
    setQuery(q);
    if (q.length > 1) {
      const res = await searchFn(q);
      setResults(res.data.actors || res.data.directors || []);
    } else {
      setResults([]);
    }
  };

  return (
    <div className="bg-gray-900 rounded-lg p-8 mb-6">
      <h2 className="text-xl font-semibold mb-4">{title}</h2>
      <input type="text" value={query} onChange={(e) => handleSearch(e.target.value)} placeholder={`${title} 검색...`} className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg mb-4" />
      {results.length > 0 && (
        <div className="max-h-40 overflow-y-auto mb-4">
          {results.map(r => <div key={r.id} onClick={() => { onAdd(r.name); setQuery(''); setResults([]); }} className="p-2 hover:bg-gray-700 cursor-pointer">{r.name}</div>)}
        </div>
      )}
      <div className="flex flex-wrap gap-2">
        {items.map(item => <div key={item} className="bg-red-600/20 text-red-300 px-3 py-1 rounded-full flex items-center gap-2">{item} <button type="button" onClick={() => onRemove(item)}>×</button></div>)}
      </div>
    </div>
  );
}
