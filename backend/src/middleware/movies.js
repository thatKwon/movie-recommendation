const mongoose = require('mongoose');
const Movie = require('../models/Movie');

async function findMovie(req, res, next) {
  const rawId = req.params.id;
  if (!rawId) {
    return res.status(400).json({ error: 'Missing id parameter' });
  }

  let movie = null;
  if (mongoose.Types.ObjectId.isValid(rawId)) {
    movie = await Movie.findById(rawId);
  } else {
    const tmdbCandidate = Number(rawId);
    if (!Number.isNaN(tmdbCandidate)) {
      movie = await Movie.findOne({ tmdbId: tmdbCandidate });
    }
  }

  if (!movie) {
    return res.status(404).json({ error: 'Movie not found' });
  }

  req.movie = movie;
  next();
}

module.exports = { findMovie };
