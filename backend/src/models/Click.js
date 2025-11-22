const mongoose = require('mongoose');

const clickSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  movieId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Movie',
    required: true
  },
  clickedAt: {
    type: Date,
    default: Date.now
  }
});

// Create indexes
clickSchema.index({ userId: 1 });
clickSchema.index({ movieId: 1 });
clickSchema.index({ userId: 1, clickedAt: -1 });

module.exports = mongoose.model('Click', clickSchema);
