const express = require('express');
const router = express.Router();
const Actor = require('../models/Actor');
const Movie = require('../models/Movie');
const Like = require('../models/Like');
const { optionalAuth } = require('../middleware/auth');
/**
 * GET /api/actors/search
 * Search actors by name and return brief filmography
 * Auth: Optional
 */
router.get('/search', optionalAuth, async (req, res) => {
  try {
    const q = (req.query.q || '').toString().trim();
    if (!q) return res.status(400).json({ error: 'Query required' });

    // text search with fallback to case-insensitive regex
    let actors = await Actor.find({ $text: { $search: q } }).limit(10);
    if (!actors || actors.length === 0) {
      actors = await Actor.find({ name: { $regex: q, $options: 'i' } }).limit(10);
    }

    const results = [];
    for (const actor of actors) {
      const movies = await Movie.find({ _id: { $in: actor.movieIds } })
        .sort({ year: -1 })
        .limit(8);
      results.push({
        id: actor._id,
        name: actor.name,
        nameEnglish: actor.nameEnglish,
        profileUrl: actor.profileUrl,
        movies: movies.map(m => ({ id: m._id, title: m.title, year: m.year, posterUrl: m.posterUrl }))
      });
    }

    res.status(200).json({ actors: results });
  } catch (error) {
    console.error('Search actors error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * GET /api/actors/:id
 * Get actor details and filmography
 * Auth: Optional
 */
router.get('/:id', optionalAuth, async (req, res) => {
  try {
    const actorId = req.params.id;

    // Find actor
    const actor = await Actor.findById(actorId);

    if (!actor) {
      return res.status(404).json({ error: 'Actor not found' });
    }

    // Check if user liked this actor (if authenticated)
    let userLiked = false;
    if (req.user) {
      const like = await Like.findOne({
        userId: req.user.userId,
        targetType: 'actor',
        targetId: actor._id
      });
      userLiked = !!like;
    }

    // Get movies where actor appeared
    const movies = await Movie.find({
      'cast.actorId': actor._id
    }).sort({ year: -1 });

    // Format movie data with character names
    const filmography = movies.map(movie => {
      const castMember = movie.cast.find(c => c.actorId.toString() === actor._id.toString());
      return {
        id: movie._id,
        title: movie.title,
        year: movie.year,
        posterUrl: movie.posterUrl,
        character: castMember ? castMember.character : null
      };
    });

    res.status(200).json({
      actor: {
        id: actor._id,
        tmdbId: actor.tmdbId,
        name: actor.name,
        nameEnglish: actor.nameEnglish,
        profileUrl: actor.profileUrl,
        likeCount: actor.likeCount,
        userLiked,
        movies: filmography
      }
    });
  } catch (error) {
    console.error('Get actor error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

module.exports = router;
