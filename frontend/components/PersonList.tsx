'use client';

import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import Image from 'next/image';
import { useAuth } from '@/context/AuthContext';
import { likesAPI } from '@/lib/api';

type ActorItem = {
    actorId: string;
    actorName: string;
    character?: string | null;
    profileUrl?: string;
};

type DirectorItem = {
    directorId: string;
    directorName: string;
    profileUrl?: string;
};

type Props = {
    type: 'Actor' | 'Director';
    items: ActorItem[] | DirectorItem[];
};

export default function PersonList({ type, items }: Props) {
    const { isAuthenticated } = useAuth();
    const router = useRouter();
    const [likedMap, setLikedMap] = useState<Record<string, boolean>>({});
    const [loadingIds, setLoadingIds] = useState<Record<string, boolean>>({});

    const normalized = useMemo(
        () =>
            (items || [])
                .map((it: any) =>
                    type === 'Actor'
                        ? {
                            id: String(it.actorId),
                            name: it.actorName,
                            sub: it.character ?? null,
                            img: it.profileUrl || null
                        }
                        : {
                            id: String(it.directorId),
                            name: it.directorName,
                            sub: null,
                            img: it.profileUrl || null
                        }
                )
                .filter(item => item.id && item.id !== 'undefined'),
        [items, type]
    );

    useEffect(() => {
        if (!isAuthenticated || normalized.length === 0) return;

        const validItems = normalized.filter(n => n.id);
        if (validItems.length === 0) return;

        const payload = validItems.map((n) => ({ type, id: n.id }));

        likesAPI
            .check(payload)
            .then((res) => {
                const map: Record<string, boolean> = {};
                const liked = res.data?.liked || {};
                for (const n of validItems) {
                    map[n.id] = Boolean(liked[`${type}_${n.id}`]);
                }
                setLikedMap(map);
            })
            .catch(() => {});
    }, [isAuthenticated, normalized, type]);

    const toggle = async (id: string) => {
        if (!isAuthenticated) {
            router.push('/login');
            return;
        }
        if (loadingIds[id]) return;
        setLoadingIds((s) => ({ ...s, [id]: true }));
        try {
            const isLiked = likedMap[id];
            if (isLiked) {
                await likesAPI.deleteByTarget(type, id);
            } else {
                await likesAPI.create(type, id);
            }
            setLikedMap((s) => ({ ...s, [id]: !isLiked }));
        } catch (e) {
            // ignore
        } finally {
            setLoadingIds((s) => ({ ...s, [id]: false }));
        }
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {normalized.map((n) => (
                <div key={n.id} className="flex items-center justify-between bg-white/5 rounded-lg p-3 hover:bg-white/10 transition border border-white/5">

                    <Link href={type === 'Actor' ? `/actor/${n.id}` : `/director/${n.id}`} className="flex items-center gap-3 min-w-0 flex-1 group">
                        {/* Circular Image */}
                        <div className="relative w-12 h-12 flex-shrink-0 rounded-full overflow-hidden bg-gray-700 ring-2 ring-white/10 group-hover:ring-red-500 transition">
                            {n.img ? (
                                <Image
                                    src={n.img}
                                    alt={n.name}
                                    fill
                                    className="object-cover"
                                    sizes="48px"
                                />
                            ) : (
                                <div className="w-full h-full flex items-center justify-center text-gray-400 bg-gray-800 text-lg font-bold">
                                    {n.name.charAt(0).toUpperCase()}
                                </div>
                            )}
                        </div>

                        {/* Name & Character */}
                        <div className="min-w-0">
                            <div className="font-medium text-gray-100 truncate group-hover:text-red-500 transition" title={n.name}>
                                {n.name}
                            </div>
                            {n.sub && (
                                <div className="text-xs text-gray-400 truncate" title={n.sub}>
                                    {n.sub}
                                </div>
                            )}
                        </div>
                    </Link>

                    {/* Like Button */}
                    <button
                        onClick={() => toggle(n.id)}
                        disabled={Boolean(loadingIds[n.id])}
                        className={`ml-3 p-2 rounded-full transition-all flex-shrink-0 ${
                            likedMap[n.id]
                                ? 'bg-red-600 text-white hover:bg-red-700'
                                : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'
                        }`}
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 24 24"
                            fill={likedMap[n.id] ? 'currentColor' : 'none'}
                            stroke="currentColor"
                            className="w-5 h-5"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.737-.715-1.611-2.377-2.737-4.313-2.737C5.099 3.75 3 5.765 3 8.25c0 7.125 9 12 9 12s9-4.875 9-12z"
                            />
                        </svg>
                    </button>
                </div>
            ))}
        </div>
    );
}