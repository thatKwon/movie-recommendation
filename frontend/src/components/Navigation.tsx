import { Home, Search, User, Film, LogOut } from 'lucide-react';

interface NavigationProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  onLogout: () => void;
  darkMode?: boolean;
}

export function Navigation({ currentPage, onNavigate, onLogout, darkMode = true }: NavigationProps) {
  const navItems = [
    { id: 'home', label: '홈', icon: Home },
    { id: 'search', label: '검색', icon: Search },
    { id: 'user', label: '마이페이지', icon: User }
  ];

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 backdrop-blur-sm border-b ${
      darkMode 
        ? 'bg-black/95 border-zinc-800' 
        : 'bg-white/95 border-zinc-300'
    }`}>
      <div className="max-w-7xl mx-auto px-4 md:px-8">
        <div className="flex items-center justify-between h-16">
          <div 
            className="flex items-center gap-2 cursor-pointer"
            onClick={() => onNavigate('home')}
          >
            <Film className="h-8 w-8 text-red-600" />
            <span className={`text-xl ${darkMode ? 'text-white' : 'text-zinc-900'}`}>MovieFlix</span>
          </div>

          <div className="flex items-center gap-1 md:gap-4">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentPage === item.id;
              
              return (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                    darkMode
                      ? isActive 
                        ? 'bg-zinc-800 text-white' 
                        : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50'
                      : isActive
                        ? 'bg-zinc-200 text-zinc-900'
                        : 'text-zinc-600 hover:text-zinc-900 hover:bg-zinc-100'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span className="hidden sm:inline">{item.label}</span>
                </button>
              );
            })}
            
            <button
              onClick={onLogout}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                darkMode
                  ? 'text-zinc-400 hover:text-white hover:bg-zinc-800/50'
                  : 'text-zinc-600 hover:text-zinc-900 hover:bg-zinc-100'
              }`}
            >
              <LogOut className="h-5 w-5" />
              <span className="hidden sm:inline">로그아웃</span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
