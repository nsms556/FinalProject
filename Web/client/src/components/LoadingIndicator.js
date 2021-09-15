import {Alert} from "reactstrap";
import {usePromiseTracker} from "react-promise-tracker"


export const LoadingIndicator = () => {

    const {promiseInProgress} = usePromiseTracker();

    return (
        promiseInProgress &&
        <Alert color="secondary">
            추천받은 곡을 확인하는 중입니다.<br/>
            잠시만 기다려주세요. 😉
        </Alert>
    );
};

export default LoadingIndicator;
