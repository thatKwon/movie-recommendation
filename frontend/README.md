cd ../frontend
echo "# MovieFlix Frontend

The client-side application for MovieFlix, a personalized movie recommendation platform. Built with Next.js (App Router), TypeScript, and Tailwind CSS.

## ✨ Features

- **User Authentication:** Secure Login/Signup flow.
- **Personalized Dashboard (My Page):** View statistics and manage liked Movies, Actors, and Directors.
- **Movie Discovery:** Browsing carousels for New Releases, Trending, and Genre-based recommendations.
- **Search:** Real-time movie search functionality.
- **Details View:** Comprehensive pages for Movies (including Cast/Director info).
- **Interactive UI:** Like/Unlike functionality with custom confirmation modals and Dark Mode support.

## 🛠 Tech Stack

- **Framework:** Next.js 13+ (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **State Management:** React Context API (AuthContext, ThemeContext)
- **HTTP Client:** Axios

## 🚀 Getting Started

### 1. Prerequisites
- Node.js (v14 or higher)
- The **Backend Server** must be running on port 5001.

### 2. Installation

1. Navigate to the frontend directory:
   \`\`\`bash
   cd frontend
   \`\`\`

2. Install dependencies:
   \`\`\`bash
   npm install
   \`\`\`

### 3. Environment Configuration

Create a \`.env.local\` file in the \`frontend\` root directory:

\`\`\`env
# URL of the backend server
NEXT_PUBLIC_API_URL=http://localhost:5001
\`\`\`

### 4. Configuration for Images

To display movie posters and actor profiles from TMDB, ensure your \`next.config.mjs\` (or \`.js\`) is configured as follows:

\`\`\`javascript
const nextConfig = {
images: {
remotePatterns: [
{
protocol: 'https',
hostname: 'image.tmdb.org',
pathname: '/**',
},
],
},
};

export default nextConfig;
\`\`\`

### 5. Running the Application

Run the development server:
\`\`\`bash
npm run dev
\`\`\`

Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

## 📂 Key Directory Structure

- **\`app/\`**: Next.js App Router pages.
   - **\`page.tsx\`**: The main landing page.
   - **\`movie/[id]/\`**: Dynamic movie detail page.
   - **\`user/\`**: The User Profile / My Page.
- **\`components/\`**: Reusable UI components (e.g., \`Carousel\`, \`MovieActions\`, \`PersonList\`, \`Header\`).
- **\`context/\`**: Global state providers (\`AuthContext\`, \`ThemeContext\`).
- **\`lib/\`**: API configuration and Axios setup (\`api.ts\`)." 