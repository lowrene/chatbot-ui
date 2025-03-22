import { ThemeToggle } from "./theme-toggle";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";

export const Header = () => {
  const handleRefresh = () => {
    window.location.reload();
  };

  return (
    <>
      <header className="flex items-center justify-between px-2 sm:px-4 py-2 bg-background text-black dark:text-white w-full">
        <div className="flex items-center space-x-1 sm:space-x-2">
          <ThemeToggle />
        </div>
      </header>
      <header className="flex items-center justify-between px-2 sm:px-4 py-2 bg-background text-black dark:text-white w-full">
        <div className="flex items-center space-x-1 sm:space-x-2">
          <Button
            variant="outline"
            className="bg-background border border-gray text-gray-600 hover:white dark:text-gray-200 h-10"
            onClick={handleRefresh}
          >
            <Trash2 className="h-[1.2rem] w-[1.2rem]" />
            <span className="sr-only">Refresh</span>
          </Button>
        </div>
      </header>
    </>
  );
};
