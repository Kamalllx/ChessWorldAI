import { useNavigate } from "react-router-dom";
import SidebarButton from "./sidebarButton";
import Icon from "./icon";

const HomeButton = () => {
  const navigate = useNavigate();

  const handleClick = () => {    
    window.location.reload(); // Refresh page since home is now the upload page
  }
  
  return (
    <SidebarButton onClick={handleClick} >
      <Icon iconName="bi-arrow-clockwise"/> Reset
    </SidebarButton>
  );
};

export default HomeButton;