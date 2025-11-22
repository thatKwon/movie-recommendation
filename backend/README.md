cd backend
echo "# MovieFlix Backend API

The backend service for MovieFlix, built with Node.js, Express, and MongoDB. It handles user authentication, data persistence (likes/preferences), and communicates with the TMDB API to fetch movie data.

## 🛠 Tech Stack

- **Runtime:** Node.js
- **Framework:** Express.js
- **Database:** MongoDB (via Mongoose)
- **Authentication:** JWT (Access & Refresh Tokens)
- **External API:** TMDB (The Movie Database)

## 🚀 Getting Started

### 1. Prerequisites
- Node.js (v14 or higher)
- MongoDB (Local instance or Atlas URL)
- A valid TMDB API Key

### 2. Installation

1. Navigate to the backend directory:
   \`\`\`bash
   cd backend
   \`\`\`

2. Install dependencies:
   \`\`\`bash
   npm install
   \`\`\`

### 3. Environment Configuration

Create a \`.env\` file in the \`backend\` root directory and add the following variables:

\`\`\`env
PORT=5001
MONGODB_URI=mongodb://localhost:27017/movieflix
JWT_SECRET=your_super_secret_access_key
REFRESH_TOKEN_SECRET=your_super_secret_refresh_key
TMDB_API_KEY=your_tmdb_api_key_here
\`\`\`

### 4. Running the Server

**Development Mode:**
\`\`\`bash
npm run dev
\`\`\`

**Production Start:**
\`\`\`bash
npm start
\`\`\`

The server will start at \`http://localhost:5001\`.

## 📂 Key Directory Structure

- **\`src/config\`**: Database connection and TMDB configuration.
- **\`src/models\`**: Mongoose schemas (User, Movie, Actor, Director, Like).
- **\`src/routes\`**: API endpoints (\`auth\`, \`movies\`, \`user\`, \`likes\`).
- **\`src/services\`**: Business logic (TMDB data fetching, Auth services).
- **\`src/middleware\`**: Authentication verification logic.

## 🔌 API Endpoints Overview

### Auth
- \`POST /api/auth/signup\` - Register a new user.
- \`POST /api/auth/login\` - Log in.
- \`POST /api/auth/refresh\` - Refresh access token using cookie.
- \`POST /api/auth/logout\` - Log out.

### User
- \`GET /api/user/profile\` - Get current user details.
- \`GET /api/user/stats\` - Get like counts for the profile page.
- \`PATCH /api/user/profile\` - Update user nickname or preferences.

### Movies
- \`GET /api/movies/new\` - Get new movie releases.
- \`GET /api/movies/:id\` - Get movie details (caches data from TMDB).
- \`GET /api/movies/search\` - Search for movies.

### Likes
- \`GET /api/likes\` - Get all items liked by the user (Movies, Actors, Directors).
- \`POST /api/likes\` - Like a specific target.
- \`DELETE /api/likes\` - Remove a like.

## 🛠 Utilities

**Database Reset Script:**
If you need to wipe all users and likes for a fresh start, run:
\`\`\`bash
node reset-users.js
