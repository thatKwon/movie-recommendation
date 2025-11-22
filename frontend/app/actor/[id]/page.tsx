// typescript
import { notFound } from 'next/navigation';
import BackButton from '@/components/BackButton';
import ActorMoviesGrid from '@/components/ActorMoviesGrid';

type Props = { params: Promise<{ id: string }> | { id: string } };

export default async function ActorPage(props: Props) {
  const { id } = await props.params;
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';

  const res = await fetch(`${API_BASE}/api/actors/${id}`, { cache: 'no-store' });
  if (res.status === 404) return notFound();
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Failed to fetch actor ${id}: ${res.status} ${body}`);
  }
  const data = await res.json();
  const actor = data.actor;

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <section className="container mx-auto px-4 pt-24 pb-12">
        <div className="flex items-center justify-end mb-4">
          <BackButton />
        </div>
        <div className="flex items-start gap-6">
          <div className="w-32 h-32 rounded-full bg-gray-800 overflow-hidden flex-shrink-0">
            {actor.profileUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={actor.profileUrl} alt={actor.name} className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-gray-500">No Image</div>
            )}
          </div>
          <div>
            <h1 className="text-3xl font-bold mb-1">{actor.name}</h1>
            {actor.nameEnglish && <p className="text-gray-400">{actor.nameEnglish}</p>}
          </div>
        </div>
      </section>

      {/* Filmography */}
      <section className="container mx-auto px-4 pb-16">
        <h2 className="text-2xl font-bold mb-6">출연작</h2>
        <ActorMoviesGrid movies={Array.isArray(actor.movies) ? actor.movies : []} />
      </section>
    </div>
  );
}


