// javascript
// file: `backend/src/utils/sanitizePreferences.js`
function sanitizePreferences(raw = {}) {
    const prefs = {};

    // genres -> array of strings
    prefs.genres = Array.isArray(raw.genres)
        ? raw.genres.map(g => (g && typeof g === 'object' ? String(g.name ?? g) : String(g))).filter(Boolean)
        : [];

    // actors -> array of strings (accept objects with name or plain strings)
    prefs.actors = Array.isArray(raw.actors)
        ? raw.actors.map(a => {
            if (a == null) return null;
            if (typeof a === 'object') return String(a.name ?? a.toString());
            return String(a);
        }).filter(Boolean)
        : [];

    // years -> ensure numbers, drop any Mongo _id
    if (raw.years && typeof raw.years === 'object') {
        const min = Number(raw.years.min);
        const max = Number(raw.years.max);
        prefs.years = {
            min: Number.isFinite(min) ? min : null,
            max: Number.isFinite(max) ? max : null
        };
    } else {
        prefs.years = { min: null, max: null };
    }

    // copy simple scalar preferences (no nested objects, no _id)
    for (const key of Object.keys(raw)) {
        if (['genres', 'actors', 'years', '_id'].includes(key)) continue;
        const v = raw[key];
        if (v == null) continue;
        if (typeof v === 'object') continue;
        prefs[key] = v;
    }

    return prefs;
}

module.exports = { sanitizePreferences };
