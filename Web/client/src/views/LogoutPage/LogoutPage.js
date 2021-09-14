import {useEffect} from "react";
import {useDispatch} from "react-redux";

import {logoutUser} from "../../_actions/user_actions";


const LogoutPage = () => {

    const dispatch = useDispatch();

    useEffect(() => {

        dispatch(logoutUser())
            .then((res) => {
                if (res.payload.success) {
                    window.localStorage.removeItem('username');
                    window.location.replace("/");
                } else {
                    alert("로그아웃에 실패했습니다.");
                }
            })
            .catch((err) => {
                console.log(err);
            })
    }, [dispatch]);

    return (
        <div/>
    );
};

export default LogoutPage;
